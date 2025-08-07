struct Vertex {
  const float spherical_harmonics[NUM_SPHERICAL_HARMONICS][3];
  const float signed_distance;
  const bool valid;
}

Vertex read_vertex(
    const TensorView<float> signed_distances,
    const TensorView<float> spherical_harmonics,
    const TensorView<int> occupancy,
    const TensorView<int> offsets,
    const int i,
    const int j,
    const int k,
) {
  float loaded_spherical_harmonics[NUM_SPHERICAL_HARMONICS][3];
  float loaded_signed_distance;

  // Out-of-bounds reads are automatically invalid.
  if (i < 0 || j < 0 || j < 0) {
    return { loaded_spherical_harmonics, loaded_signed_distance, false };
  }
  let k_packed = k / 32;
  if (i >= occupancy.size(0) || j >= occupancy.size(1) ||
      k_packed >= occupancy.size(2)) {
    return { loaded_spherical_harmonics, loaded_signed_distance, false };
  }

  // Check if the vertex is occupied.
  let bit = k % 32;
  var occupancy_chunk = occupancy[i, j, k_packed];
  if ((occupancy_chunk & (1 << bit)) == 0) {
    return { loaded_spherical_harmonics, loaded_signed_distance, false };
  }

  // If the vertex is occupied, find its index.
  var vertex_index = offsets[i, j, k_packed];
  for (int d = 0; d < bit; d++) {
    if ((occupancy_chunk & 1) == 1) {
      vertex_index++;
    }
    occupancy_chunk >>= 1;
  }

  // Read the vertex's spherical harmonics and signed distance.
  for (int s = 0; s < NUM_SPHERICAL_HARMONICS; s++) {
    loaded_spherical_harmonics[s][0] = spherical_harmonics[s, vertex_index, 0];
    loaded_spherical_harmonics[s][1] = spherical_harmonics[s, vertex_index, 1];
    loaded_spherical_harmonics[s][2] = spherical_harmonics[s, vertex_index, 2];
  }
  loaded_signed_distance = signed_distances[vertex_index];
  return { loaded_spherical_harmonics, loaded_signed_distance, true };
}

Vertex sample_trilinear(
    const TensorView<float> signed_distances,
    const TensorView<float> spherical_harmonics,
    const TensorView<int> occupancy,
    const TensorView<int> offsets,
    const float x,
    const float y,
    const float z,
) {
  // Compute an integer index and a fractional part for each dimension.
  let x_unnormalized = x * float((occupancy.size(2) - 1) * 32);
  let y_unnormalized = y * float(occupancy.size(1) - 1);
  let z_unnormalized = z * float(occupancy.size(0) - 1);
  let i = int(z_unnormalized);
  let j = int(y_unnormalized);
  let k = int(x_unnormalized);
  let t_i = z_unnormalized % 1;
  let t_j = y_unnormalized % 1;
  let t_k = x_unnormalized % 1;

  float out_spherical_harmonics[NUM_SPHERICAL_HARMONICS][3];
  for (int s = 0; s < NUM_SPHERICAL_HARMONICS; s++) {
    out_spherical_harmonics[s][0] = 0;
    out_spherical_harmonics[s][1] = 0;
    out_spherical_harmonics[s][2] = 0;
  }
  float out_signed_distance = 0;
  float denominator = 0;

  for (int d_i = 0; d_i < 2; d_i++) {
    for (int d_j = 0; d_j < 2; d_j++) {
      for (int d_k = 0; d_k < 2; d_k++) {
        // Read the vertex at the corner of the cube.
        let vertex = read_vertex(
            signed_distances,
            spherical_harmonics,
            occupancy,
            offsets,
            i + d_i,
            j + d_j,
            k + d_k,
        );

        // Skip past invalid vertices.
        if (!vertex.valid) {
          continue;
        }

        // Add valid vertices to the interpolation.
        let weight = (d_i == 0 ? 1 - t_i : t_i) * (d_j == 0 ? 1 - t_j : t_j) *
                     (d_k == 0 ? 1 - t_k : t_k);
        for (int s = 0; s < NUM_SPHERICAL_HARMONICS; s++) {
          out_spherical_harmonics[s][0] += weight * vertex.spherical_harmonics[s][0];
          out_spherical_harmonics[s][1] += weight * vertex.spherical_harmonics[s][1];
          out_spherical_harmonics[s][2] += weight * vertex.spherical_harmonics[s][2];
        }
        out_signed_distance += weight * vertex.signed_distance;
        denominator += weight;
      }
    }
  }

  // Normalize the result.
  let multiplier = 1 / denominator;
  for (int s = 0; s < NUM_SPHERICAL_HARMONICS; s++) {
    out_spherical_harmonics[s][0] *= multiplier;
    out_spherical_harmonics[s][1] *= multiplier;
    out_spherical_harmonics[s][2] *= multiplier;
  }
  out_signed_distance *= multiplier;

  return { out_spherical_harmonics, out_signed_distance, true };
}

[AutoPyBindCUDA]
[CUDAKernel]
void upscale_parameters(
    const TensorView<int> old_vertex_occupancy,
    const TensorView<int> old_vertex_offsets,
    const TensorView<float> old_signed_distances,
    const TensorView<float> old_spherical_harmonics,
    const TensorView<int> new_vertex_occupancy,
    const TensorView<int> new_vertex_offsets,
    const TensorView<float> new_signed_distances,
    const TensorView<float> new_spherical_harmonics,
) {
  // Read the packed occupancy grid's dimensions.
  let i_max = new_vertex_occupancy.size(0) - 1;
  let j_max = new_vertex_occupancy.size(1) - 1;
  let k_packed_max = new_vertex_occupancy.size(2) - 1;

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
  var vertex_index = new_vertex_offsets[index];
  var vertex_occupancy_chunk = new_vertex_occupancy[index];
  var k = k_unpacked;
  for (int bit = 0; bit < 32; bit++) {
    // If the least significant bit is set, the voxel is vertex-occupied.
    if ((vertex_occupancy_chunk & 1) == 1) {
      // Do trilinear interpolation of the old values and write out the result.
      let x = k * inv_k_max;
      let y = j * inv_j_max;
      let z = i * inv_i_max;
      Vertex vertex = sample_trilinear(
          old_signed_distances,
          old_spherical_harmonics,
          old_vertex_occupancy,
          old_vertex_offsets,
          x,
          y,
          z
      );
      for (int s = 0; s < NUM_SPHERICAL_HARMONICS; s++) {
        new_spherical_harmonics[s, vertex_index, 0] = vertex.spherical_harmonics[s][0];
        new_spherical_harmonics[s, vertex_index, 1] = vertex.spherical_harmonics[s][1];
        new_spherical_harmonics[s, vertex_index, 2] = vertex.spherical_harmonics[s][2];
      }
      new_signed_distances[vertex_index] = vertex.signed_distance;

      // Go to the next vertex.
      vertex_index++;
    }

    // Step to the next bit.
    vertex_occupancy_chunk >>= 1;
    k++;
  }
}
