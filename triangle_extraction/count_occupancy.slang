[AutoPyBindCUDA]
[CUDAKernel]
void count_occupancy(const TensorView<int> occupancy, TensorView<int> voxel_counts) {
  // Read the packed occupancy grid's dimensions.
  let i_max = occupancy.size(0);
  let j_max = occupancy.size(1);
  let k_packed_max = occupancy.size(2);

  // Guard against indices that are invalid in the occupancy grid.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let k_packed = index.z;
  if (i >= i_max || j >= j_max || k_packed >= k_packed_max) {
    return;
  }

  // Count the number of voxels in the current packed vector.
  voxel_counts[index] = countbits(occupancy[index]);
}
