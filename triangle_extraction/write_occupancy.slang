[AutoPyBindCUDA]
[CUDAKernel]
void write_occupancy(
    const TensorView<int> occupancy,
    const TensorView<int> voxel_indices,
    const int min_shell,
    const int max_shell,
) {
  // Guard against indices that are invalid in the new occupancy grid.
  int index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (index >= voxel_indices.size(0)) {
    return;
  }

  // Mark the triangle's location in the occupancy grid as occupied.
  let voxel_index = voxel_indices[index];
  let array_index = voxel_index >> 5;
  let bit = reinterpret<int>(1u << (voxel_index % 32));
  __atomic_or(*occupancy.data_ptr_at(array_index), bit, MemoryOrder.Relaxed);
}
