[AutoPyBindCUDA]
[CUDAKernel]
void countTriangleVertices(
    const TensorView<float> signed_distances,
    const TensorView<int> lower_corners,
    const TensorView<float> level_sets,
    const TensorView<int> vertex_counts_by_level_set,
) {
  let index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (index >= lower_corners.size(1)) {
    return;
  }

  // The order is (origin, +x, +y, +z).
  let d0 = signed_distances[lower_corners[0, index]];
  let dx =
      (lower_corners[1, index] >= 0) ? signed_distances[lower_corners[1, index]] : d0;
  let dy =
      (lower_corners[2, index] >= 0) ? signed_distances[lower_corners[2, index]] : d0;
  let dz =
      (lower_corners[3, index] >= 0) ? signed_distances[lower_corners[3, index]] : d0;

  for (int level_set = 0; level_set < level_sets.size(0); level_set++) {
    let iso = level_sets[level_set];
    int num_vertices = 0;
    if ((d0 < iso) != (dx < iso)) {
      num_vertices += 1;
    }
    if ((d0 < iso) != (dy < iso)) {
      num_vertices += 1;
    }
    if ((d0 < iso) != (dz < iso)) {
      num_vertices += 1;
    }
    vertex_counts_by_level_set[level_set, index] = num_vertices;
  }
}
