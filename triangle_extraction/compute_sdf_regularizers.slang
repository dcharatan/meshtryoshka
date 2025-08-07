static const int NUM_EIKONAL_SAMPLES = 1;

[Differentiable]
float stable_norm(const float a, const float b, const float c) {
  // This will not compile on some nodes with a let here.
  const float scale = max(max(abs(a), abs(b)), abs(c));
  if (scale < 1e-10) {
    return 0;
  }
  let invScale = 1 / scale;
  let scaledA = invScale * a;
  let scaledB = invScale * b;
  let scaledC = invScale * c;
  let sum = scaledA * scaledA + scaledB * scaledB + scaledC * scaledC;
  return scale * sqrt(sum);
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void compute_sdf_regularizers(
    const DiffTensorView grid_signed_distances,
    const TensorView<int> voxel_neighbors,
    const TensorView<int> voxel_lower_corners,
    const TensorView<int> voxel_upper_corners,
    const int voxel_x,
    const int voxel_y,
    const int voxel_z,
    const DiffTensorView eikonal_loss_pos,
    const DiffTensorView eikonal_loss_neg,
    const DiffTensorView curvature_loss,
) {
  // Skip out-of-bounds voxels.
  int index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  let num_voxels = voxel_upper_corners.size(1);
  if (index >= num_voxels) {
    return;
  }

  // Load the current voxel's signed distances.
  let d000 = grid_signed_distances[voxel_lower_corners[0, index]]; // origin
  let d001 = grid_signed_distances[voxel_lower_corners[1, index]]; // +x
  let d010 = grid_signed_distances[voxel_lower_corners[2, index]]; // +y
  let d100 = grid_signed_distances[voxel_lower_corners[3, index]]; // +z
  let d111 = grid_signed_distances[voxel_upper_corners[0, index]]; // +xyz
  let d110 = grid_signed_distances[voxel_upper_corners[1, index]]; // +yz
  let d101 = grid_signed_distances[voxel_upper_corners[2, index]]; // +xz
  let d011 = grid_signed_distances[voxel_upper_corners[3, index]]; // +xy

  // Compute the Eikonal loss.
  const float3 sample_locations[NUM_EIKONAL_SAMPLES] = {
    float3(0.5, 0.5, 0.5),
  };
  float eikonal_sum_pos = 0;
  float eikonal_sum_neg = 0;
  for (int i = 0; i < NUM_EIKONAL_SAMPLES; i++) {
    let tx = sample_locations[i][0];
    let ty = sample_locations[i][1];
    let tz = sample_locations[i][2];
    let tx1 = 1 - tx;
    let ty1 = 1 - ty;
    let tz1 = 1 - tz;

    // Compute the magnitude of the trilinearly interpolated signed distance's gradient.
    let dx = ty1 * tz1 * (d001 - d000) + ty * tz1 * (d011 - d010) +
             ty1 * tz * (d101 - d100) + ty * tz * (d111 - d110);
    let dy = tx1 * tz1 * (d010 - d000) + tx * tz1 * (d011 - d001) +
             tx1 * tz * (d110 - d100) + tx * tz * (d111 - d101);
    let dz = tx1 * ty1 * (d100 - d000) + tx * ty1 * (d101 - d001) +
             tx1 * ty * (d110 - d010) + tx * ty * (d111 - d011);

    // Compute the Eikonal loss as the square of the difference between the above
    // gradient magnitude and 1. Note that we scale the gradient magnitude by the grid
    // size.
    let delta = stable_norm(voxel_x * dx, voxel_y * dy, voxel_z * dz) - 1;
    if (delta < 0) {
      eikonal_sum_neg += delta * delta;
    }
    if (delta > 0) {
      eikonal_sum_pos += delta * delta;
    }
  }
  eikonal_loss_pos[index] = eikonal_sum_pos / NUM_EIKONAL_SAMPLES;
  eikonal_loss_neg[index] = eikonal_sum_neg / NUM_EIKONAL_SAMPLES;

  // Compute the curvature loss. The curvature loss looks like a 3D Laplacian filter:
  //
  // Top layer:
  // -- -- --
  // --  1 --
  // -- -- --
  //
  // Middle layer:
  // --  1 --
  //  1 -6  1
  // --  1 --
  //
  // Bottom layer:
  // -- -- --
  // --  1 --
  // -- -- --
  //
  // The value that corresponds to the -6 weight is the current voxel's d111 corner.
  // Three of the values that correspond to 1 weights are the current voxel's d110,
  // d101, and d011 corners. The other three values that correspond to 1 weights may be
  // absent, since they're associated with the subvoxel with offset (1, 1, 1) from the
  // current voxel. If the values are absent, we make the -6 weight smaller.

  // Initialize the curvature sum with the signed distances that are always present.
  float curvature_sum = d110 + d101 + d011;
  int num_counted = 3;

  // Add whatever signed distances are present in the neighboring subvoxel.
  let subvoxel_index = voxel_neighbors[6, index];
  for (int i = 1; i < 4; i++) {
    let subvoxel_corner_index = voxel_lower_corners[i, subvoxel_index];
    if (subvoxel_corner_index >= 0) {
      curvature_sum += grid_signed_distances[subvoxel_corner_index];
      num_counted++;
    }
  }

  // Weight the middle signed distance by the number of present signed distances.
  curvature_sum -= num_counted * d111;

  // Turn the result into an L2 loss on curvature.
  curvature_loss[index] = curvature_sum * curvature_sum;
}
