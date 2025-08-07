import "./constants.slang";

int get_cube_code(const float cube_sdf[8], const float iso) {
  int code = 0;
  for (int i = 0; i < 8; i++) {
    if (cube_sdf[i] >= iso) {
      code |= 1 << i;
    }
  }
  return code;
}

[AutoPyBindCUDA]
[CUDAKernel]
void count_triangle_faces(
    const TensorView<float> signed_distances,
    const TensorView<int> lower_corners,
    const TensorView<int> upper_corners,
    const TensorView<float> level_sets,
    TensorView<int> face_counts,
    TensorView<uint8_t> voxel_cell_codes,
) {
  int index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (index >= upper_corners.size(1)) {
    return;
  }

  // We arrange the corners to match cumc.cu's mcCorners.
  const float cube_sdf[8] = {
    signed_distances[lower_corners[0, index]], // origin
    signed_distances[lower_corners[1, index]], // x
    signed_distances[upper_corners[3, index]], // xy
    signed_distances[lower_corners[2, index]], // y
    signed_distances[lower_corners[3, index]], // z
    signed_distances[upper_corners[2, index]], // xy
    signed_distances[upper_corners[0, index]], // xyz
    signed_distances[upper_corners[1, index]], // yz
  };

  for (int level_set = 0; level_set < level_sets.size(0); level_set++) {
    let iso = level_sets[level_set];

    let cube_code = get_cube_code(cube_sdf, iso);
    voxel_cell_codes[level_set, index] = uint8_t(cube_code);

    let numFaces =
        FIRST_MARCHING_CUBES_ID[cube_code + 1] - FIRST_MARCHING_CUBES_ID[cube_code];
    face_counts[level_set, index] = numFaces;
  }
}
