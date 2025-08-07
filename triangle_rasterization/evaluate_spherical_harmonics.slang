// Adapted from code by George Kopanas.
import "./common.slang";

static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;
static const float SH_C2[] = {
  1.0925484305920792f,
  -1.0925484305920792f,
  0.31539156525252005f,
  -1.0925484305920792f,
  0.5462742152960396f
};
static const float SH_C3[] = {
  -0.5900435899266435f,
  2.890611442640554f,
  -0.4570457994644658f,
  0.3731763325901154f,
  -0.4570457994644658f,
  1.445305721320277f,
  -0.5900435899266435f
};

static const int NUM_SH_COEFFICIENTS = (SH_DEGREE + 1) * (SH_DEGREE + 1);
typealias SphericalHarmonics = float3[NUM_SH_COEFFICIENTS];

[Differentiable]
SphericalHarmonics load_spherical_harmonics(
    const uint32_t vertex_index,
    const DiffTensorView spherical_harmonics,
) {
  SphericalHarmonics coefficients;
  coefficients[0] = float3(
      load(spherical_harmonics, uint3(0, vertex_index, 0)),
      load(spherical_harmonics, uint3(0, vertex_index, 1)),
      load(spherical_harmonics, uint3(0, vertex_index, 2))
  );

  if (SH_DEGREE > 0) {
    coefficients[1] = float3(
        load(spherical_harmonics, uint3(1, vertex_index, 0)),
        load(spherical_harmonics, uint3(1, vertex_index, 1)),
        load(spherical_harmonics, uint3(1, vertex_index, 2))
    );
    coefficients[2] = float3(
        load(spherical_harmonics, uint3(2, vertex_index, 0)),
        load(spherical_harmonics, uint3(2, vertex_index, 1)),
        load(spherical_harmonics, uint3(2, vertex_index, 2))
    );
    coefficients[3] = float3(
        load(spherical_harmonics, uint3(3, vertex_index, 0)),
        load(spherical_harmonics, uint3(3, vertex_index, 1)),
        load(spherical_harmonics, uint3(3, vertex_index, 2))
    );

    if (SH_DEGREE > 1) {
      coefficients[4] = float3(
          load(spherical_harmonics, uint3(4, vertex_index, 0)),
          load(spherical_harmonics, uint3(4, vertex_index, 1)),
          load(spherical_harmonics, uint3(4, vertex_index, 2))
      );
      coefficients[5] = float3(
          load(spherical_harmonics, uint3(5, vertex_index, 0)),
          load(spherical_harmonics, uint3(5, vertex_index, 1)),
          load(spherical_harmonics, uint3(5, vertex_index, 2))
      );
      coefficients[6] = float3(
          load(spherical_harmonics, uint3(6, vertex_index, 0)),
          load(spherical_harmonics, uint3(6, vertex_index, 1)),
          load(spherical_harmonics, uint3(6, vertex_index, 2))
      );
      coefficients[7] = float3(
          load(spherical_harmonics, uint3(7, vertex_index, 0)),
          load(spherical_harmonics, uint3(7, vertex_index, 1)),
          load(spherical_harmonics, uint3(7, vertex_index, 2))
      );
      coefficients[8] = float3(
          load(spherical_harmonics, uint3(8, vertex_index, 0)),
          load(spherical_harmonics, uint3(8, vertex_index, 1)),
          load(spherical_harmonics, uint3(8, vertex_index, 2))
      );

      if (SH_DEGREE > 2) {
        coefficients[9] = float3(
            load(spherical_harmonics, uint3(9, vertex_index, 0)),
            load(spherical_harmonics, uint3(9, vertex_index, 1)),
            load(spherical_harmonics, uint3(9, vertex_index, 2))
        );
        coefficients[10] = float3(
            load(spherical_harmonics, uint3(10, vertex_index, 0)),
            load(spherical_harmonics, uint3(10, vertex_index, 1)),
            load(spherical_harmonics, uint3(10, vertex_index, 2))
        );
        coefficients[11] = float3(
            load(spherical_harmonics, uint3(11, vertex_index, 0)),
            load(spherical_harmonics, uint3(11, vertex_index, 1)),
            load(spherical_harmonics, uint3(11, vertex_index, 2))
        );
        coefficients[12] = float3(
            load(spherical_harmonics, uint3(12, vertex_index, 0)),
            load(spherical_harmonics, uint3(12, vertex_index, 1)),
            load(spherical_harmonics, uint3(12, vertex_index, 2))
        );
        coefficients[13] = float3(
            load(spherical_harmonics, uint3(13, vertex_index, 0)),
            load(spherical_harmonics, uint3(13, vertex_index, 1)),
            load(spherical_harmonics, uint3(13, vertex_index, 2))
        );
        coefficients[14] = float3(
            load(spherical_harmonics, uint3(14, vertex_index, 0)),
            load(spherical_harmonics, uint3(14, vertex_index, 1)),
            load(spherical_harmonics, uint3(14, vertex_index, 2))
        );
        coefficients[15] = float3(
            load(spherical_harmonics, uint3(15, vertex_index, 0)),
            load(spherical_harmonics, uint3(15, vertex_index, 1)),
            load(spherical_harmonics, uint3(15, vertex_index, 2))
        );
      }
    }
  }
  return coefficients;
}

[Differentiable]
float3 evaluate_spherical_harmonics(
    const SphericalHarmonics sh,
    const float3 vertex_position,
    const float3 camera_position,
) {
  let dir = normalize(vertex_position - camera_position);

  float3 rgb = SH_C0 * sh[0];
  if (SH_DEGREE > 0) {
    rgb = rgb - SH_C1 * dir.y * sh[1] + SH_C1 * dir.z * sh[2] - SH_C1 * dir.x * sh[3];
    if (SH_DEGREE > 1) {
      let xx = dir.x * dir.x;
      let yy = dir.y * dir.y;
      let zz = dir.z * dir.z;
      let xy = dir.x * dir.y;
      let yz = dir.y * dir.z;
      let xz = dir.x * dir.z;
      rgb = rgb + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] +
            SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] + SH_C2[3] * xz * sh[7] +
            SH_C2[4] * (xx - yy) * sh[8];

      if (SH_DEGREE > 2) {
        rgb = rgb + SH_C3[0] * dir.y * (3.0f * xx - yy) * sh[9] +
              SH_C3[1] * xy * dir.z * sh[10] +
              SH_C3[2] * dir.y * (4.0f * zz - xx - yy) * sh[11] +
              SH_C3[3] * dir.z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
              SH_C3[4] * dir.x * (4.0f * zz - xx - yy) * sh[13] +
              SH_C3[5] * dir.z * (xx - yy) * sh[14] +
              SH_C3[6] * dir.x * (xx - 3.0f * yy) * sh[15];
      }
    }
  }

  return max(rgb + 0.5, 0.0);
}

float3 load_camera_position(
    const TensorView<float> camera_positions,
    const uint camera_index,
) {
  return float3(
      camera_positions[camera_index, 0],
      camera_positions[camera_index, 1],
      camera_positions[camera_index, 2],
  );
}

[Differentiable]
void evaluate_spherical_harmonics_for_camera(
    const uint vertex_index,
    const uint camera_index,
    const no_diff float3 camera_position,
    const no_diff float3 vertex,
    const SphericalHarmonics spherical_harmonics,
    const DiffTensorView out_colors,
) {
  let rgb = evaluate_spherical_harmonics(spherical_harmonics, vertex, camera_position);
  store_once(out_colors, uint3(camera_index, vertex_index, 0), rgb.r);
  store_once(out_colors, uint3(camera_index, vertex_index, 1), rgb.g);
  store_once(out_colors, uint3(camera_index, vertex_index, 2), rgb.b);
}

[AutoPyBindCUDA]
[CUDAKernel]
void evaluate_spherical_harmonics_forward(
    const TensorView<float> vertices,
    const DiffTensorView spherical_harmonics,
    const TensorView<float> camera_positions,
    const DiffTensorView out_colors,
) {
  // Skip past invalid vertices.
  let vertex_index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (vertex_index >= vertices.size(0)) {
    return;
  }

  // Load per-vertex information.
  let coefficients = load_spherical_harmonics(vertex_index, spherical_harmonics);
  let vertex = float3(
      vertices[vertex_index, 0],
      vertices[vertex_index, 1],
      vertices[vertex_index, 2],
  );

  // Loop through the cameras.
  let num_cameras = camera_positions.size(0);
  for (uint camera_index = 0; camera_index < num_cameras; camera_index++) {
    let camera_position = load_camera_position(camera_positions, camera_index);
    evaluate_spherical_harmonics_for_camera(
        vertex_index,
        camera_index,
        camera_position,
        vertex,
        coefficients,
        out_colors,
    );
  }
}

[AutoPyBindCUDA]
[CUDAKernel]
void evaluate_spherical_harmonics_backward(
    const TensorView<float> vertices,
    const DiffTensorView spherical_harmonics,
    const TensorView<float> camera_positions,
    const DiffTensorView out_colors,
) {
  // Skip past invalid vertices.
  let vertex_index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (vertex_index >= vertices.size(0)) {
    return;
  }

  // Load per-vertex information.
  let coefficients = load_spherical_harmonics(vertex_index, spherical_harmonics);
  let vertex = float3(
      vertices[vertex_index, 0],
      vertices[vertex_index, 1],
      vertices[vertex_index, 2],
  );

  // Loop through the cameras.
  let num_cameras = camera_positions.size(0);
  SphericalHarmonics.Differential d_coefficients = SphericalHarmonics.dzero();
  for (uint camera_index = 0; camera_index < num_cameras; camera_index++) {
    // Backpropagate through the individual camera's spherical harmonics evaluation.
    let camera_position = load_camera_position(camera_positions, camera_index);
    var dp_coefficients = diffPair(coefficients);
    bwd_diff(evaluate_spherical_harmonics_for_camera)(
        vertex_index,
        camera_index,
        camera_position,
        vertex,
        dp_coefficients,
        out_colors,
    );

    // Accumulate gradients across loop iterations.
    for (uint i = 0; i < NUM_SH_COEFFICIENTS; i++) {
      d_coefficients[i] += dp_coefficients.d[i];
    }
  }

  // Store the accumulated gradients.
  bwd_diff(load_spherical_harmonics)(vertex_index, spherical_harmonics, d_coefficients);
}
