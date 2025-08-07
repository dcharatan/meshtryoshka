import "./common.slang";
import "./keys.slang";

[AutoPyBindCUDA]
[CUDAKernel]
void generate_keys(
    const TensorView<float> depths,
    const TensorView<int> faces,
    const TensorView<int> shell_face_boundaries,
    const TensorView<int> tile_minima,
    const TensorView<int> tile_maxima,
    const TensorView<int64_t> offsets,
    const int grid_width,
    const TensorView<int64_t> out_keys,
    const TensorView<int> out_triangle_indices,
) {
  // Parallelize over faces.
  const int face_index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  const int num_faces = faces.size(0);
  if (face_index >= num_faces) {
    return;
  }

  // Determine the shell index for this face.
  uint shell_index = 0;
  for (; shell_index < shell_face_boundaries.size(0); shell_index++) {
    if (face_index < shell_face_boundaries[shell_index]) {
      break;
    }
  }

  // Load the vertex indices for this face.
  let vertex_a = load(faces, uint64_t2(face_index, 0));
  let vertex_b = load(faces, uint64_t2(face_index, 1));
  let vertex_c = load(faces, uint64_t2(face_index, 2));

  // Loop through the cameras.
  const uint num_cameras = tile_minima.size(0);
  for (uint camera_index = 0; camera_index < num_cameras; camera_index++) {
    let offset_index = (camera_index * num_faces) + face_index;
    var key_index = (offset_index == 0) ? 0 : load(offsets, offset_index - 1);

    // Retrieve the tile overlap bounds.
    const int row_min = load(tile_minima, uint64_t3(camera_index, face_index, 0));
    const int col_min = load(tile_minima, uint64_t3(camera_index, face_index, 1));
    const int row_max = load(tile_maxima, uint64_t3(camera_index, face_index, 0));
    const int col_max = load(tile_maxima, uint64_t3(camera_index, face_index, 1));

    // Compute the face's mean depth. It's not necessary to filter out special-case
    // triangles behind the camera because they were already filtered out during the
    // bound_triangles step of rasterization.
    let depth_a = load(depths, uint64_t2(camera_index, vertex_a));
    let depth_b = load(depths, uint64_t2(camera_index, vertex_b));
    let depth_c = load(depths, uint64_t2(camera_index, vertex_c));
    let mean_depth = (depth_a + depth_b + depth_c) / 3.f;

    // Write out a key for each overlapped tile.
    for (uint row = row_min; row < row_max; row++) {
      for (uint col = col_min; col < col_max; col++) {
        let tile_index = row * grid_width + col;
        let key = pack_key(camera_index, shell_index, tile_index, mean_depth);
        store(out_keys, key_index, reinterpret<int64_t>(key));
        store(out_triangle_indices, key_index, face_index);
        key_index++;
      }
    }
  }
}
