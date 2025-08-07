int logical_right_shift(const int x, const int shift) {
  // In Slang, int uses arithmetic right shift, and uint uses logical right shift. Thus,
  // to get a logical right shift for int, we have to cast to uint and back.
  return reinterpret<int>(reinterpret<uint>(x) >> shift);
}

[AutoPyBindCUDA]
[CUDAKernel]
void compute_vertex_occupancy(
    const TensorView<int> occupancy,
    TensorView<int> vertex_occupancy,
    TensorView<int> vertex_counts,
) {
  // Read the packed occupancy grid's dimensions.
  let i_max = occupancy.size(0);
  let j_max = occupancy.size(1);
  let k_packed_max = occupancy.size(2);

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

  // A particular voxel is considered vertex-occupied if it or any of its lower
  // neighbors are occupied. Along the i and j axes, we simply compare packed vectors
  // to the corresponding packed vectors that are one index lower. Along the k axis,
  // we compare the packed vectors to versions of themselves that are shifted by one
  // bit. We fill each shifted packed vector's missing bit using the most significant
  // bit from the previous packed vector.
  int result = 0;
  for (int di = 0; di < 2; di++) {
    for (int dj = 0; dj < 2; dj++) {
      // Handle the low edge case along the i and j axes.
      if ((i == 0 && di == 1) || (j == 0 && dj == 1)) {
        continue;
      }

      // Handle the high edge case along the i and j axes.
      if ((i == i_max && di == 0) || (j == j_max && dj == 0)) {
        continue;
      }

      // Fetch the packed vector at the current index.
      let current =
          (k_packed == k_packed_max) ? 0 : occupancy[i - di, j - dj, k_packed];

      // Fetch the packed vector at the previous index along the k axis.
      let previous_packed =
          (k_packed == 0) ? 0 : occupancy[i - di, j - dj, k_packed - 1];

      // Compare packed vectors along the k axis as described above.
      let previous = logical_right_shift(previous_packed, 31) | (current << 1);
      result |= (current | previous);
    }
  }

  // Set the vertex occupancy.
  vertex_occupancy[index] = result;

  // Set the number of vertices owned by the group of packed voxels.
  vertex_counts[index] = countbits(reinterpret<uint>(result));
}
