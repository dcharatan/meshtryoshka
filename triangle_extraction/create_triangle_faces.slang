import "./constants.slang";

[AutoPyBindCUDA]
[CUDAKernel]
void create_triangle_faces(
    const TensorView<float> signed_distances,
    const TensorView<int> neighbors,
    const TensorView<int> indices,
    const TensorView<int> vertex_offsets,
    const TensorView<uint8_t> triangle_vertex_types,
    const TensorView<uint8_t> voxel_cell_codes,
    const TensorView<float> level_sets,
    const TensorView<int> triangle_faces,
    const TensorView<int> voxel_indices,
    const int num_voxels_and_subvoxels,
    const TensorView<int> face_offsets,
) {
  let index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  let num_voxels = indices.size(0);
  if (index >= num_voxels) {
    return;
  }

  for (int level_set = 0; level_set < level_sets.size(0); level_set++) {
    // This value is 3 * num_tris, so it is total number of vertices to add to our
    // buffer
    let faceOffset = (level_set * num_voxels) + index;
    let num = face_offsets[faceOffset + 1] - face_offsets[faceOffset];
    let triangle_write_ctr = face_offsets[faceOffset];

    const int code = voxel_cell_codes[level_set, index];
    let num_i = FIRST_MARCHING_CUBES_ID[code + 1] - FIRST_MARCHING_CUBES_ID[code];
    let firstIn = FIRST_MARCHING_CUBES_ID[code];
    for (int j = 0; j < num_i; j++) {
      // Edge id
      let eid = MARCHING_CUBES_IDS[firstIn + j];

      // Which neighbor owns this edge?
      let neighbor_index = MC_EDGE_LOCATIONS[eid][0];

      // Basically the direction of the edge
      let edge_nr = MC_EDGE_LOCATIONS[eid][1];

      int neighbor_voxel_index;
      if (neighbor_index == -1) {
        neighbor_voxel_index = index;
      } else {
        neighbor_voxel_index = neighbors[neighbor_index, index];
      }

      // For that neighbor, we need to access the triangle vertices it owns
      let vertex_offset_index =
          (level_set * num_voxels_and_subvoxels) + neighbor_voxel_index;
      let start_idx = vertex_offsets[vertex_offset_index];
      int neighbor_num_vertices = vertex_offsets[vertex_offset_index + 1] - start_idx;

      // Now we are going to check all triangles vertices (with the right level set)
      // owned by the neighbor, and find the unique one that is the same orientation!
      int target_vertex = -1;
      for (int k = 0; k < neighbor_num_vertices; k++) {
        let query_vertex_idx = start_idx + k;
        if (triangle_vertex_types[query_vertex_idx] == edge_nr) {
          target_vertex = query_vertex_idx;
          break;
        }
      }

      // Now write the vertex idx to triangle faces buffer
      // We need to convert from 3*num_tris to num_tris, 3
      let triangle_num = (triangle_write_ctr + j) / 3;
      let triangle_offset = 2 - (triangle_write_ctr + j) % 3;
      triangle_faces[triangle_num, triangle_offset] = target_vertex;
      if (triangle_offset == 0) {
        voxel_indices[triangle_num] = indices[index];
      }
    }
  }
}
