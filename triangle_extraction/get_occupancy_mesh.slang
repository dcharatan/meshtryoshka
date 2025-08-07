[AutoPyBindCUDA]
[CUDAKernel]
void count_triangles(
    const TensorView<int> occupancy,
    const TensorView<int> triangle_counts,
) {
  // Read the packed occupancy grid's dimensions.
  let i_max = occupancy.size(0);
  let j_max = occupancy.size(1);
  let k_packed_max = occupancy.size(2);

  // Guard against invalid indices. Note that indices where i == i_max, j == j_max, or
  // k == k_packed_max are considered valid, since we special-case these below.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let k_packed = index.z;
  if (i > i_max || j > j_max || k_packed > k_packed_max) {
    return;
  }

  // Fetch occupancy from the current packed vector and its neighbors.
  let i_lim = (i == i_max);
  let j_lim = (j == j_max);
  let k_lim = (k_packed == k_packed_max);
  var at_here = (i_lim || j_lim || k_lim) ? 0 : occupancy[i, j, k_packed];
  var at_minus_i = (j_lim || k_lim || (i == 0)) ? 0 : occupancy[i - 1, j, k_packed];
  var at_minus_j = (i_lim || k_lim || (j == 0)) ? 0 : occupancy[i, j - 1, k_packed];
  var bit_at_minus_k = (i_lim || j_lim || (k_packed == 0))
                           ? false
                           : ((occupancy[i, j, k_packed - 1] & (1 << 31)) != 0);

  // Iterate through the packed bits and count triangles.
  int num_faces = 0;
  for (int bit = 0; bit < 32; bit++) {
    // Extract the current bit.
    let bit_at_here = (at_here & 1) != 0;
    let bit_at_minus_i = (at_minus_i & 1) != 0;
    let bit_at_minus_j = (at_minus_j & 1) != 0;

    // Count the faces created by the current voxel.
    num_faces += int(bit_at_here != bit_at_minus_i);
    num_faces += int(bit_at_here != bit_at_minus_j);
    num_faces += int(bit_at_here != bit_at_minus_k);

    // Walk to the next bit.
    bit_at_minus_k = bit_at_here;
    at_here >>= 1;
    at_minus_i >>= 1;
    at_minus_j >>= 1;
  }

  triangle_counts[i, j, k_packed] = 2 * num_faces;
}

void create_face(
    const int index,
    const int ijk[3],
    const int ijk_max[3],
    const int axis,
    const bool upper_bit,
    const TensorView<float> triangle_vertices,
    const TensorView<float> triangle_colors,
    const TensorView<float> colors,
) {
  let axis0 = (axis + 1) % 3;
  let axis1 = (axis + 2) % 3;

  // Define the vertex locations.
  float3 v00;
  float3 v01;
  float3 v10;
  float3 v11;

  let plane = float(ijk[axis]) / ijk_max[axis];
  v00[axis] = plane;
  v01[axis] = plane;
  v10[axis] = plane;
  v11[axis] = plane;

  let v0x = float(ijk[axis0]) / ijk_max[axis0];
  let v1x = float(ijk[axis0] + 1) / ijk_max[axis0];
  let vx0 = float(ijk[axis1]) / ijk_max[axis1];
  let vx1 = float(ijk[axis1] + 1) / ijk_max[axis1];

  v00[axis0] = v0x;
  v00[axis1] = vx0;

  v01[axis0] = v0x;
  v01[axis1] = vx1;

  v10[axis0] = v1x;
  v10[axis1] = vx0;

  v11[axis0] = v1x;
  v11[axis1] = vx1;

  // Determine the face color.
  let side = int(upper_bit);
  let face_color =
      float3(colors[axis, side, 0], colors[axis, side, 1], colors[axis, side, 2]);

  // Write out the resulting triangles.
  const float3 triangles[2][3] = {
    { v00, upper_bit ? v01 : v10, upper_bit ? v10 : v01 },
    { v11, upper_bit ? v10 : v01, upper_bit ? v01 : v10 },
  };
  $for(i in Range(0, 2)) {
    let offset = index + i;
    $for(j in Range(0, 3)) {
      // The triangles are indexed by axis (ijk) but need to be flipped to xyz.
      triangle_vertices[offset, j, 0] = triangles[i][j].z;
      triangle_vertices[offset, j, 1] = triangles[i][j].y;
      triangle_vertices[offset, j, 2] = triangles[i][j].x;
    }
    triangle_colors[offset, 0] = face_color.r;
    triangle_colors[offset, 1] = face_color.g;
    triangle_colors[offset, 2] = face_color.b;
  }
}

[AutoPyBindCUDA]
[CUDAKernel]
void create_triangles(
    const TensorView<int> occupancy,
    const TensorView<int> triangle_offsets,
    const TensorView<float> triangle_vertices,
    const TensorView<float> triangle_colors,
    const TensorView<float> colors,
) {
  // Read the packed occupancy grid's dimensions.
  let i_max = occupancy.size(0);
  let j_max = occupancy.size(1);
  let k_packed_max = occupancy.size(2);

  // Guard against invalid indices. Note that indices where i == i_max, j == j_max, or
  // k == k_packed_max are considered valid, since we special-case these below.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let k_packed = index.z;
  if (i > i_max || j > j_max || k_packed > k_packed_max) {
    return;
  }

  // Fetch occupancy from the current packed vector and its neighbors.
  let i_lim = (i == i_max);
  let j_lim = (j == j_max);
  let k_lim = (k_packed == k_packed_max);
  var at_here = (i_lim || j_lim || k_lim) ? 0 : occupancy[i, j, k_packed];
  var at_minus_i = (j_lim || k_lim || (i == 0)) ? 0 : occupancy[i - 1, j, k_packed];
  var at_minus_j = (i_lim || k_lim || (j == 0)) ? 0 : occupancy[i, j - 1, k_packed];
  var bit_at_minus_k = (i_lim || j_lim || (k_packed == 0))
                           ? false
                           : ((occupancy[i, j, k_packed - 1] & (1 << 31)) != 0);

  // Iterate through the packed bits and count triangles.
  int triangle_index = triangle_offsets[i, j, k_packed];
  int k = k_packed * 32;
  const int ijk_max[3] = { i_max, j_max, k_packed_max * 32 };
  for (int bit = 0; bit < 32; bit++) {
    // Extract the current bit.
    let bit_at_here = (at_here & 1) != 0;
    let bit_at_minus_i = (at_minus_i & 1) != 0;
    let bit_at_minus_j = (at_minus_j & 1) != 0;
    const int ijk[3] = { i, j, k };

    // Create faces with Z-aligned normals.
    if (bit_at_here != bit_at_minus_i) {
      create_face(
          triangle_index,
          ijk,
          ijk_max,
          0,
          bit_at_here,
          triangle_vertices,
          triangle_colors,
          colors,
      );
      triangle_index += 2;
    }

    // Create faces with Y-aligned normals.
    if (bit_at_here != bit_at_minus_j) {
      create_face(
          triangle_index,
          ijk,
          ijk_max,
          1,
          bit_at_here,
          triangle_vertices,
          triangle_colors,
          colors,
      );
      triangle_index += 2;
    }

    // Create faces with X-aligned normals.
    if (bit_at_here != bit_at_minus_k) {
      create_face(
          triangle_index,
          ijk,
          ijk_max,
          2,
          bit_at_here,
          triangle_vertices,
          triangle_colors,
          colors,
      );
      triangle_index += 2;
    }

    // Walk to the next bit.
    bit_at_minus_k = bit_at_here;
    at_here >>= 1;
    at_minus_i >>= 1;
    at_minus_j >>= 1;
    k++;
  }
}
