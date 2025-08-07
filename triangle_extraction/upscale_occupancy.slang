bool sample_corner(
    const TensorView<int> occupancy,
    const int i,
    const int j,
    const int k,
    const int i_max,
    const int j_max,
    const int k_max,
    const int i_side,
    const int j_side,
    const int k_side,
) {
  // The coordinates are integers but represent corners. We use integer arithmetic to
  // sample the lower-resolution occupancy grid to avoid rounding problems.
  let i_max_sampled = occupancy.size(0);
  let j_max_sampled = occupancy.size(1);
  let k_max_sampled = occupancy.size(2) << 5;
  let i_sampled = ((i + i_side) * i_max_sampled - i_side) / i_max;
  let j_sampled = ((j + j_side) * j_max_sampled - j_side) / j_max;
  let k_sampled = ((k + k_side) * k_max_sampled - k_side) / k_max;
  let bit = reinterpret<int>(1u << (k_sampled % 32));
  return (occupancy[i_sampled, j_sampled, k_sampled >> 5] & bit) != 0;
}

bool sample_voxel(
    const TensorView<int> occupancy,
    const int i,
    const int j,
    const int k,
    const int i_max,
    const int j_max,
    const int k_max,
) {
  bool value = false;
  for (int di = 0; di < 2; di++) {
    for (int dj = 0; dj < 2; dj++) {
      for (int dk = 0; dk < 2; dk++) {
        value |= sample_corner(occupancy, i, j, k, i_max, j_max, k_max, di, dj, dk);
      }
    }
  }
  return value;
}

[AutoPyBindCUDA]
[CUDAKernel]
void upscale_occupancy(
    const TensorView<int> occupancy,
    const TensorView<int> new_occupancy,
) {
  // Read the packed new occupancy grid's dimensions.
  let i_max = new_occupancy.size(0);
  let j_max = new_occupancy.size(1);
  let k_packed_max = new_occupancy.size(2);

  // Guard against indices that are invalid in the new occupancy grid.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let k_packed = index.z;
  if (i >= i_max || j >= j_max || k_packed >= k_packed_max) {
    return;
  }
  let k_max = k_packed_max << 5;
  let k_base = k_packed << 5;

  // Work through the bits in this thread's entry in the new occupancy grid one by one.
  int result_packed = 0;
  for (int bit = 0; bit < 32; bit++) {
    let result_bit = sample_voxel(occupancy, i, j, k_base + bit, i_max, j_max, k_max);
    result_packed |= (int(result_bit) << bit);
  }
  new_occupancy[i, j, k_packed] = result_packed;
}
