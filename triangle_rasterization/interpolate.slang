import "./common.slang";

[Differentiable]
float4 load_corner(
    const DiffTensorView signed_distances,
    const DiffTensorView colors,
    const uint batch_index,
    const uint vertex_index,
) {
  let signed_distance = load(signed_distances, vertex_index);
  let color = float3(
      load(colors, uint3(batch_index, vertex_index, 0)),
      load(colors, uint3(batch_index, vertex_index, 1)),
      load(colors, uint3(batch_index, vertex_index, 2)),
  );
  return float4(color, signed_distance);
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void interpolate(
    const TensorView<float> uv,
    const TensorView<int> index,
    const TensorView<int> faces,
    const DiffTensorView signed_distances,
    const DiffTensorView colors,
    const DiffTensorView out_signed_distances,
    const DiffTensorView out_colors,
    const int num_shells,
) {
  // Guard against out-of-bounds threads.
  let bs_index = cudaBlockIdx().z;
  let batch_index = bs_index / num_shells;
  let row = cudaBlockIdx().y * cudaBlockDim().y + cudaThreadIdx().y;
  let col = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  let height = uv.size(1);
  let width = uv.size(2);
  if (row >= height || col >= width) {
    return;
  }

  // Load the pixel's barycentric coordinates.
  let loaded_u = no_diff load(uv, uint4(bs_index, row, col, 0));
  let loaded_v = no_diff load(uv, uint4(bs_index, row, col, 1));
  let uvw = no_diff float3(loaded_u, loaded_v, 1 - loaded_u - loaded_v);

  // Handle the case where there is no triangle.
  let face_index = load(index, uint3(bs_index, row, col));
  if (face_index < 0) {
    store_once(out_colors, uint4(bs_index, row, col, 0), 0);
    store_once(out_colors, uint4(bs_index, row, col, 1), 0);
    store_once(out_colors, uint4(bs_index, row, col, 2), 0);
    store_once(out_signed_distances, uint3(bs_index, row, col), 0);
    return;
  }

  // Load the vertex indices for the pixel.
  let vertex_index_a = load(faces, uint2(face_index, 0));
  let vertex_index_b = load(faces, uint2(face_index, 1));
  let vertex_index_c = load(faces, uint2(face_index, 2));

  // Load the corner values.
  let a = load_corner(signed_distances, colors, batch_index, vertex_index_a);
  let b = load_corner(signed_distances, colors, batch_index, vertex_index_b);
  let c = load_corner(signed_distances, colors, batch_index, vertex_index_c);

  // Interpolate the corner values.
  let result = uvw.x * a + uvw.y * b + uvw.z * c;

  // Store the interpolated signed distance and color.
  store_once(out_colors, uint4(bs_index, row, col, 0), result.x);
  store_once(out_colors, uint4(bs_index, row, col, 1), result.y);
  store_once(out_colors, uint4(bs_index, row, col, 2), result.z);
  store_once(out_signed_distances, uint3(bs_index, row, col), result.w);
}
