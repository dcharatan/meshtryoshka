[AutoPyBindCUDA]
[CUDAKernel]
void create_voxels(
    const TensorView<int> voxel_occupancy,
    const TensorView<int> voxel_offsets,
    const TensorView<int> vertex_occupancy,
    const TensorView<int> vertex_offsets,
    const TensorView<float> vertices,
    const TensorView<int> neighbors,
    const TensorView<int> lower_corners,
    const TensorView<int> upper_corners,
    const TensorView<int> indices,
) {
  // Read the packed occupancy grid's dimensions.
  let i_max = voxel_occupancy.size(0);
  let j_max = voxel_occupancy.size(1);
  let k_packed_max = voxel_occupancy.size(2);

  // Guard against indices that are invalid in the vertex occupancy grid. Note that
  // the highest indices in the vertex occupancy grid are out of bounds in the
  // occupancy grid.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let k_packed = index.z;
  if (i > i_max || j > j_max || k_packed > k_packed_max) {
    return;
  }

  // Do some math that will be needed to set vertex positions later.
  let k_unpacked = 32 * k_packed;
  let inv_i_max = 1 / float(i_max);
  let inv_j_max = 1 / float(j_max);
  let inv_k_max = 1 / float(32 * k_packed_max);

  // Allocate vertices.
  var vertex_index = vertex_offsets[index];
  var vertex_occupancy_chunk = vertex_occupancy[index];
  var k = k_unpacked;
  for (int bit = 0; bit < 32; bit++) {
    // If the least significant bit is set, the voxel is vertex-occupied.
    if ((vertex_occupancy_chunk & 1) == 1) {
      // Allocate the vertex.
      vertices[vertex_index, 0] = k * inv_k_max;
      vertices[vertex_index, 1] = j * inv_j_max;
      vertices[vertex_index, 2] = i * inv_i_max;
      vertex_index++;
    }

    // Step to the next bit.
    vertex_occupancy_chunk >>= 1;
    k++;
  }

  // Allocate voxels and subvoxels.
  let num_voxels = neighbors.size(1);
  int local_voxel_indices[2][2];
  int local_voxel_occupancies[2][2];
  int local_vertex_indices[2][2];
  int local_vertex_occupancies[2][2];
  int local_subvoxel_indices[2][2];
  int local_subvoxel_occupancies[2][2];
  for (int di = 0; di < 2; di++) {
    for (int dj = 0; dj < 2; dj++) {
      let ii = i + di;
      let jj = j + dj;
      let local_index = uint3(ii, jj, k_packed);

      // Handle out-of-bounds indices.
      if (ii > i_max || jj > j_max) {
        local_voxel_indices[di][dj] = -64;
        local_voxel_occupancies[di][dj] = 0;
        local_vertex_indices[di][dj] = -64;
        local_vertex_occupancies[di][dj] = 0;
        local_subvoxel_indices[di][dj] = -64;
        local_subvoxel_occupancies[di][dj] = 0;
        continue;
      }

      let valid_voxel = (ii < i_max && jj < j_max && k_packed < k_packed_max);
      let local_voxel_index = voxel_offsets[local_index];
      let local_voxel_occupancy = valid_voxel ? voxel_occupancy[local_index] : 0;
      let local_vertex_index = vertex_offsets[local_index];
      let local_vertex_occupancy = vertex_occupancy[local_index];

      local_voxel_indices[di][dj] = local_voxel_index;
      local_voxel_occupancies[di][dj] = local_voxel_occupancy;
      local_vertex_indices[di][dj] = local_vertex_index;
      local_vertex_occupancies[di][dj] = local_vertex_occupancy;
      local_subvoxel_indices[di][dj] =
          num_voxels + local_vertex_index - local_voxel_index;
      local_subvoxel_occupancies[di][dj] =
          local_vertex_occupancy & ~local_voxel_occupancy;
    }
  }

  let k_max = k_packed_max * 32;
  k = k_unpacked;
  for (int bit = 0; bit < 32; bit++) {
    let is_voxel = (local_voxel_occupancies[0][0] & 1) != 0;
    let is_subvoxel = (local_subvoxel_occupancies[0][0] & 1) != 0;
    let voxel_index = local_voxel_indices[0][0];

    // Set the lower corners. The order is (origin, +x, +y, +z). Note that for
    // subvoxels, the indices can be negative (invalid).
    if (is_voxel || is_subvoxel) {
      let chosen_index = is_voxel ? voxel_index : local_subvoxel_indices[0][0];

      // The origin is easy, since it's always defined.
      lower_corners[0, chosen_index] = local_vertex_indices[0][0];

      // The other corners have to be handled carefully for subvoxels because they may
      // be invalid. Here, the vertex occupancy is used as a proxy for whether the
      // adjacent entry is either a voxel or a subvoxel. If so, the corner is valid.
      // Note that the last bit has to be special-cased for subvoxels. This happens
      // later, outside the loop.
      let x_valid = is_voxel || ((local_vertex_occupancies[0][0] & 2) != 0);
      let y_valid = is_voxel || ((local_vertex_occupancies[0][1] & 1) != 0);
      let z_valid = is_voxel || ((local_vertex_occupancies[1][0] & 1) != 0);
      lower_corners[1, chosen_index] = x_valid ? local_vertex_indices[0][0] + 1 : -64;
      lower_corners[2, chosen_index] = y_valid ? local_vertex_indices[0][1] : -64;
      lower_corners[3, chosen_index] = z_valid ? local_vertex_indices[1][0] : -64;
    }

    // Set the upper corners. The order is (+xyz, +yz, +xz, +xy).
    if (is_voxel) {
      upper_corners[0, voxel_index] = local_vertex_indices[1][1] + 1;
      upper_corners[1, voxel_index] = local_vertex_indices[1][1];
      upper_corners[2, voxel_index] = local_vertex_indices[1][0] + 1;
      upper_corners[3, voxel_index] = local_vertex_indices[0][1] + 1;
      indices[voxel_index] = k_max * (j_max * i + j) + k;
    }

    // Set the neighbors. The order is (+x, +y, +z, +xy, +yz, +xz). Note that the last
    // bit has to be special-cased because it depends on the next packed occupancy.
    if (is_voxel && bit != 31) {
      // Set neighbor 0 (+x).
      let x_is_subvoxel = (local_subvoxel_occupancies[0][0] & 2) != 0;
      neighbors[0, voxel_index] =
          x_is_subvoxel ? local_subvoxel_indices[0][0] : local_voxel_indices[0][0] + 1;

      // Set neighbor 1 (+y).
      let y_is_subvoxel = (local_subvoxel_occupancies[0][1] & 1) != 0;
      neighbors[1, voxel_index] =
          y_is_subvoxel ? local_subvoxel_indices[0][1] : local_voxel_indices[0][1];

      // Set neighbor 2 (+z).
      let z_is_subvoxel = (local_subvoxel_occupancies[1][0] & 1) != 0;
      neighbors[2, voxel_index] =
          z_is_subvoxel ? local_subvoxel_indices[1][0] : local_voxel_indices[1][0];

      // Set neighbor 3 (+xy).
      let xy_is_subvoxel = (local_subvoxel_occupancies[0][1] & 2) != 0;
      neighbors[3, voxel_index] =
          xy_is_subvoxel ? (y_is_subvoxel ? local_subvoxel_indices[0][1] + 1
                                          : local_subvoxel_indices[0][1])
                         : (y_is_subvoxel ? local_voxel_indices[0][1]
                                          : local_voxel_indices[0][1] + 1);

      // Set neighbor 4 (+yz).
      let yz_is_subvoxel = (local_subvoxel_occupancies[1][1] & 1) != 0;
      neighbors[4, voxel_index] =
          yz_is_subvoxel ? local_subvoxel_indices[1][1] : local_voxel_indices[1][1];

      // Set neighbor 5 (+xz).
      let xz_is_subvoxel = (local_subvoxel_occupancies[1][0] & 2) != 0;
      neighbors[5, voxel_index] =
          xz_is_subvoxel ? (z_is_subvoxel ? local_subvoxel_indices[1][0] + 1
                                          : local_subvoxel_indices[1][0])
                         : (z_is_subvoxel ? local_voxel_indices[1][0]
                                          : local_voxel_indices[1][0] + 1);

      // Set neighbor 6 (+xyz).
      let xyz_is_subvoxel = (local_subvoxel_occupancies[1][1] & 2) != 0;
      neighbors[6, voxel_index] =
          xyz_is_subvoxel ? (yz_is_subvoxel ? local_subvoxel_indices[1][1] + 1
                                            : local_subvoxel_indices[1][1])
                          : (yz_is_subvoxel ? local_voxel_indices[1][1]
                                            : local_voxel_indices[1][1] + 1);
    }

    // Step to the next bit.
    if (bit != 31) {
      for (int di = 0; di < 2; di++) {
        for (int dj = 0; dj < 2; dj++) {
          local_voxel_indices[di][dj] += (local_voxel_occupancies[di][dj] & 1);
          local_voxel_occupancies[di][dj] >>= 1;

          local_vertex_indices[di][dj] += (local_vertex_occupancies[di][dj] & 1);
          local_vertex_occupancies[di][dj] >>= 1;

          local_subvoxel_indices[di][dj] += (local_subvoxel_occupancies[di][dj] & 1);
          local_subvoxel_occupancies[di][dj] >>= 1;
        }
      }
    }

    k++;
  }

  // Handle +x neighbors for subvoxels at the last bit.
  if ((local_subvoxel_occupancies[0][0] & 1) != 0) {
    let x_valid =
        (k_packed < k_packed_max) && ((vertex_occupancy[i, j, k_packed + 1] & 1) != 0);
    lower_corners[1, local_subvoxel_indices[0][0]] =
        x_valid ? local_vertex_indices[0][0] + 1 : -64;
  }

  // Set the neighbors for the last bit in the case where it represents a voxel. This
  // requires a special case because it relies on the next packed occupancy vector.
  // The order is again (+x, +y, +z, +xy, +yz, +xz).
  if ((local_voxel_occupancies[0][0] & 1) == 0) {
    return;
  }

  let voxel_index = local_voxel_indices[0][0];
  let can_increment_x = (k_packed < k_packed_max - 1);
  let can_increment_y = (j < j_max - 1);
  let can_increment_z = (i < i_max - 1);

  // Set neighbor 0 (+x).
  let x_is_subvoxel =
      can_increment_x ? ((voxel_occupancy[i, j, k_packed + 1] & 1) == 0) : true;
  neighbors[0, voxel_index] =
      x_is_subvoxel ? local_subvoxel_indices[0][0] : local_voxel_indices[0][0] + 1;

  // Set neighbor 1 (+y).
  let y_is_subvoxel =
      can_increment_y ? ((voxel_occupancy[i, j + 1, k_packed] & (1 << 31)) == 0) : true;
  neighbors[1, voxel_index] =
      y_is_subvoxel ? local_subvoxel_indices[0][1] : local_voxel_indices[0][1];

  // Set neighbor 2 (+z).
  let z_is_subvoxel =
      can_increment_z ? ((voxel_occupancy[i + 1, j, k_packed] & (1 << 31)) == 0) : true;
  neighbors[2, voxel_index] =
      z_is_subvoxel ? local_subvoxel_indices[1][0] : local_voxel_indices[1][0];

  // Set neighbor 3 (+xy).
  let xy_is_subvoxel = (can_increment_x && can_increment_y)
                           ? ((voxel_occupancy[i, j + 1, k_packed + 1] & 1) == 0)
                           : true;
  neighbors[3, voxel_index] =
      xy_is_subvoxel
          ? (y_is_subvoxel ? local_subvoxel_indices[0][1] + 1
                           : local_subvoxel_indices[0][1])
          : (y_is_subvoxel ? local_voxel_indices[0][1] : local_voxel_indices[0][1] + 1);

  // Set neighbor 4 (+yz).
  let yz_is_subvoxel =
      (can_increment_y && can_increment_z)
          ? ((voxel_occupancy[i + 1, j + 1, k_packed] & (1 << 31)) == 0)
          : true;
  neighbors[4, voxel_index] =
      yz_is_subvoxel ? local_subvoxel_indices[1][1] : local_voxel_indices[1][1];

  // Set neighbor 5 (+xz).
  let xz_is_subvoxel = (can_increment_x && can_increment_z)
                           ? ((voxel_occupancy[i + 1, j, k_packed + 1] & 1) == 0)
                           : true;
  neighbors[5, voxel_index] =
      xz_is_subvoxel
          ? (z_is_subvoxel ? local_subvoxel_indices[1][0] + 1
                           : local_subvoxel_indices[1][0])
          : (z_is_subvoxel ? local_voxel_indices[1][0] : local_voxel_indices[1][0] + 1);

  // Set neighbor 6 (+xyz).
  let xyz_is_subvoxel = (can_increment_x && can_increment_y && can_increment_z)
                            ? ((voxel_occupancy[i + 1, j + 1, k_packed + 1] & 1) == 0)
                            : true;
  neighbors[6, voxel_index] = xyz_is_subvoxel
                                  ? (yz_is_subvoxel ? local_subvoxel_indices[1][1] + 1
                                                    : local_subvoxel_indices[1][1])
                                  : (yz_is_subvoxel ? local_voxel_indices[1][1]
                                                    : local_voxel_indices[1][1] + 1);
}
