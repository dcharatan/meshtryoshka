static const float EPSILON = 1e-6;

[Differentiable]
float3 load_color(
    const DiffTensorView colors,
    const int pixel_index,
    const int shell_index,
) {
  return {
    colors[pixel_index, shell_index, 0],
    colors[pixel_index, shell_index, 1],
    colors[pixel_index, shell_index, 2],
  };
}

[Differentiable]
float load_sdf(
    const DiffTensorView signed_distances,
    const int pixel_index,
    const int shell_index,
) {
  return signed_distances[pixel_index, shell_index];
}

[Differentiable]
float load_sharpness(const DiffTensorView sharpness) {
  return sharpness[0];
}

[Differentiable]
float sdf_to_cdf(const float sdf, const float sharpness) {
  return 1 / (1 + exp(-sdf * sharpness));
}

[Differentiable]
float compute_alpha(
    const float inner_sdf,
    const float outer_sdf,
    const float sharpness,
) {
  let inner_cdf = sdf_to_cdf(inner_sdf, sharpness);
  let outer_cdf = sdf_to_cdf(outer_sdf, sharpness);
  return clamp((outer_cdf - inner_cdf + 1e-5) / (outer_cdf + 1e-5), 0, 1);
}

[Differentiable]
void store_result(
    const DiffTensorView out_colors,
    const DiffTensorView out_transmittances,
    const float3 color,
    const float transmittance,
    const int pixel_index,
) {
  out_colors[pixel_index, 0] = color.r;
  out_colors[pixel_index, 1] = color.g;
  out_colors[pixel_index, 2] = color.b;
  out_transmittances[pixel_index] = transmittance;
}

[AutoPyBindCUDA]
[CUDAKernel]
void composite_forward(
    const DiffTensorView signed_distances,
    const DiffTensorView colors,
    const TensorView<bool> mask,
    const DiffTensorView sharpness,
    const DiffTensorView out_colors,
    const DiffTensorView out_transmittances,
) {
  // Skip past out-of-bounds pixels.
  const int pixel_index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  const int num_pixels = signed_distances.size(0);
  if (pixel_index >= num_pixels) {
    return;
  }

  // Initialize the bookkeeping values for compositing.
  float3 color = { 0, 0, 0 };
  float transmittance = 1;
  float outer_sdf = 1e10; // essentially infinity

  // Iterate through the shells.
  let loaded_sharpness = load_sharpness(sharpness);
  const int num_shells = signed_distances.size(1);
  for (int shell_index = 0; shell_index < num_shells; shell_index++) {
    // Skip shells without a triangle intersection.
    if (!mask[pixel_index, shell_index]) {
      continue;
    }

    // Update the transmittance and color based on the current shell.
    let inner_sdf = load_sdf(signed_distances, pixel_index, shell_index);
    let inner_color = load_color(colors, pixel_index, shell_index);
    let alpha = compute_alpha(inner_sdf, outer_sdf, loaded_sharpness);
    color = color + inner_color * transmittance * alpha;
    transmittance = transmittance * (1 - alpha);

    // Update the bookkeeping values.
    outer_sdf = inner_sdf;
  }

  // Composite the result onto the background and save it.
  store_result(out_colors, out_transmittances, color, transmittance, pixel_index);
}

[AutoPyBindCUDA]
[CUDAKernel]
void composite_backward(
    const DiffTensorView signed_distances,
    const DiffTensorView colors,
    const TensorView<bool> mask,
    const DiffTensorView sharpness,
    const DiffTensorView out_colors,
    const DiffTensorView out_transmittances,
) {
  // Skip past out-of-bounds pixels.
  const int pixel_index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  const int num_pixels = signed_distances.size(0);
  if (pixel_index >= num_pixels) {
    return;
  }

  // Initialize the bookkeeping values for compositing.
  float3 color = { 0, 0, 0 };
  float transmittance = 1;
  float outer_sdf = 1e10; // essentially infinity
  bool first_shell = true;
  float3 d_color_partial = float3(0, 0, 0);
  float d_sdf_partial = 0;
  float d_sharpness = 0;

  // Load the forward pass's outputs and their gradients.
  let final_color = float3(
      out_colors[pixel_index, 0],
      out_colors[pixel_index, 1],
      out_colors[pixel_index, 2],
  );
  let final_transmittance = out_transmittances[pixel_index];
  var dp_final_color = diffPair(final_color);
  var dp_final_transmittance = diffPair(final_transmittance);
  bwd_diff(store_result)(
      out_colors,
      out_transmittances,
      dp_final_color,
      dp_final_transmittance,
      pixel_index,
  );

  // Iterate through the shells.
  let loaded_sharpness = load_sharpness(sharpness);
  const int num_shells = signed_distances.size(1);
  for (int shell_index = 0; shell_index < num_shells; shell_index++) {
    // Skip shells without a triangle intersection.
    if (!mask[pixel_index, shell_index]) {
      continue;
    }

    // Compute the current (inner) shell's color.
    let inner_sdf = load_sdf(signed_distances, pixel_index, shell_index);
    let inner_color = load_color(colors, pixel_index, shell_index);
    let alpha = compute_alpha(inner_sdf, outer_sdf, loaded_sharpness);

    // Use the classic suffix trick to compute gradients for this layer's color and
    // alpha values. Note that d_color corresponds to the inner (current) shell.
    let weight = alpha * transmittance;
    color += weight * inner_color;
    let suffix = final_color - color;
    let inverse_alpha = 1 / (1 - alpha + EPSILON);
    let d_color = weight * dp_final_color.d;
    let d_alpha_from_color =
        dot(dp_final_color.d, transmittance * inner_color - suffix * inverse_alpha);
    let d_alpha_from_transmittance =
        -dp_final_transmittance.d * final_transmittance * inverse_alpha;
    let d_alpha = d_alpha_from_color + d_alpha_from_transmittance;

    // Backpropagate through computing the current (inner) shell's alpha value. Since
    // this value depends on both the current (inner) shell's signed distance and the
    // previous (outer) shell's signed distance, we get gradients for both shells. This
    // means that each shell receives signed distance gradients during two successive
    // loop iterations. We therefore use d_sdf_partial to accumulate gradients between
    // iterations and only save a shell's gradients after the second loop iteration.
    var dp_inner_sdf = diffPair(inner_sdf);
    var dp_outer_sdf = diffPair(outer_sdf);
    var dp_sharpness = diffPair(loaded_sharpness);
    bwd_diff(compute_alpha)(dp_inner_sdf, dp_outer_sdf, dp_sharpness, d_alpha);

    // The sharpness gradient is accumulated across all loop iterations.
    d_sharpness += dp_sharpness.d;

    // The current (inner) shell's color is unaffected by the previous (outer) shell's
    // color, so we can store its gradients directly.
    bwd_diff(load_color)(colors, pixel_index, shell_index, d_color);

    // Store the previous (outer) shell's signed distance gradient.
    if (!first_shell) {
      d_sdf_partial += dp_outer_sdf.d;
      bwd_diff(load_sdf)(signed_distances, pixel_index, shell_index - 1, d_sdf_partial);
    }

    // Accumulate the current (inner) shell's signed distance gradient. It will be
    // stored once the current (inner) shell has acted as an outer shell during the next
    // loop iteration.
    d_sdf_partial = dp_inner_sdf.d;

    // Update the bookkeeping values.
    transmittance *= 1 - alpha;
    outer_sdf = inner_sdf;
    first_shell = false;
  }

  // Store signed distance gradients for the last shell.
  if (!first_shell) {
    bwd_diff(load_sdf)(signed_distances, pixel_index, num_shells - 1, d_sdf_partial);
  }

  // Store the accumulated sharpness gradient.
  bwd_diff(load_sharpness)(sharpness, d_sharpness);
}
