import "./common.slang";
import "./keys.slang";
static const float EPSILON = 1e-6;

struct Vertex : IDifferentiable {
  no_diff float2 position;
  float3 color;
  float signed_distance;
}

struct Triangle : IDifferentiable {
  int shell;
  Vertex a;
  Vertex b;
  Vertex c;
};

// Define shared memory used during the forward and backward passes.
static const uint NUM_SHARED_TRIANGLES = TILE_HEIGHT * TILE_WIDTH;
groupshared Triangle shared_triangles[NUM_SHARED_TRIANGLES];
groupshared int shared_triangle_indices[NUM_SHARED_TRIANGLES];

[Differentiable]
Vertex load_vertex(
    const TensorView<float> vertices,
    const DiffTensorView colors,
    const DiffTensorView signed_distances,
    const TensorView<int> faces,
    const uint camera_index,
    const uint face_index,
    const uint vertex_index_in_face,
) {
  let vertex_index = load(faces, uint2(face_index, vertex_index_in_face));
  let position = float2(
      no_diff load(vertices, uint3(camera_index, vertex_index, 0)),
      no_diff load(vertices, uint3(camera_index, vertex_index, 1)),
  );
  let color = float3(
      load(colors, uint3(camera_index, vertex_index, 0)),
      load(colors, uint3(camera_index, vertex_index, 1)),
      load(colors, uint3(camera_index, vertex_index, 2)),
  );
  let signed_distance = signed_distances[vertex_index];
  return { position, color, signed_distance };
}

[Differentiable]
Triangle load_triangle(
    const TensorView<float> vertices,
    const DiffTensorView colors,
    const DiffTensorView signed_distances,
    const TensorView<int> faces,
    const uint camera_index,
    const uint shell_index,
    const uint face_index,
) {
  return {
    shell_index,
    load_vertex(vertices, colors, signed_distances, faces, camera_index, face_index, 0),
    load_vertex(vertices, colors, signed_distances, faces, camera_index, face_index, 1),
    load_vertex(vertices, colors, signed_distances, faces, camera_index, face_index, 2),
  };
}

[Differentiable]
float3
compute_barycentric_coordinates(const Triangle triangle, no_diff const float2 point) {
  let p1 = triangle.a.position;
  let p2 = triangle.b.position;
  let p3 = triangle.c.position;

  // Compute dot products
  let d_12_12 = dot(p2 - p1, p2 - p1);
  let d_12_23 = dot(p2 - p1, p3 - p2);
  let d_12_13 = dot(p2 - p1, p3 - p1);
  let d_13_13 = dot(p3 - p1, p3 - p1);
  let d_13_23 = dot(p3 - p1, p3 - p2);
  let d_23_23 = dot(p3 - p2, p3 - p2);
  let d_1x_12 = dot(point - p1, p2 - p1);
  let d_1x_13 = dot(point - p1, p3 - p1);
  let d_2x_12 = dot(point - p2, p2 - p1);
  let d_2x_13 = dot(point - p2, p3 - p1);

  // Compute barycentric coordinates
  // The most stable estimate of the area of the triangle is whatever uses
  // the smallest line segment as the thing to be squared (I think).
  float denom;
  if (abs(d_12_13) <= abs(d_12_23) && abs(d_12_13) <= abs(d_13_23)) {
    denom = d_12_12 * d_13_13 - d_12_13 * d_12_13;
  } else if (abs(d_12_23) <= abs(d_12_13) && abs(d_12_23) <= abs(d_13_23)) {
    denom = d_12_12 * d_23_23 - d_12_23 * d_12_23;
  } else if (abs(d_13_23) <= abs(d_12_13) && abs(d_13_23) <= abs(d_12_23)) {
    denom = d_13_13 * d_23_23 - d_13_23 * d_13_23;
  }

  float v = d_13_13 * d_1x_12 - d_12_13 * d_1x_13;
  float w = d_12_12 * d_1x_13 - d_12_13 * d_1x_12;
  float u = d_12_23 * d_2x_13 - d_13_23 * d_2x_12;

  let eps = 1e-8;
  v /= denom + eps;
  w /= denom + eps;
  u /= denom + eps;

  return { u, v, w };
}

// This gets created when you evaluate a triangle at a point.
struct Sample : IDifferentiable {
  float3 rgb; // color
  float sdf;  // signed distance
  bool hit;   // whether the triangle was intersected
}

[Differentiable]
Sample sample_triangle(
    const Triangle triangle,
    no_diff const float2 point,
    const int last_shell,
) {
  // First, check the triangle's shell index to see if it should be skipped.
  if (triangle.shell <= last_shell) {
    return { float3(0.f, 0.f, 0.f), 0.f, false };
  }

  // Compute barycentric coordinates for the triangle.
  let uvw = compute_barycentric_coordinates(triangle, point);

  // If the barycentric coordinates land outside the triangle or appear to be
  // degenerate, return full transparency.
  let uvwSum = uvw.x + uvw.y + uvw.z;
  if (isnan(uvw.x) || isnan(uvw.y) || isnan(uvw.z) || uvw.x < 0 || uvw.x > 1 ||
      uvw.y < 0 || uvw.y > 1 || uvw.z < 0 || uvw.z > 1 || uvwSum < 0.999 ||
      uvwSum > 1.001) {
    return { float3(0.f, 0.f, 0.f), 0.f, false };
  }

  // Interpolate color and signed distance from the vertices.
  let signed_distance = uvw.x * triangle.a.signed_distance +
                        uvw.y * triangle.b.signed_distance +
                        uvw.z * triangle.c.signed_distance;
  let rgb =
      uvw.x * triangle.a.color + uvw.y * triangle.b.color + uvw.z * triangle.c.color;
  return { rgb, signed_distance, true };
}

// This is used to accumulate visibility.
struct Accumulator : IDifferentiable {
  float3 rgb;
  float transmittance;
}

[Differentiable]
float sdf_to_cdf(const float sdf, const float sharpness) {
  return 1 / (1 + exp(-sdf * sharpness));
}

[Differentiable]
float compute_alpha(const Sample inner, const Sample outer, const float sharpness) {
  let innerCdf = sdf_to_cdf(inner.sdf, sharpness);
  let outerCdf = sdf_to_cdf(outer.sdf, sharpness);
  return clamp((outerCdf - innerCdf + 1e-5) / (outerCdf + 1e-5), 0, 1);
}

[Differentiable]
Accumulator accumulate(
    const Accumulator accumulator,
    const Sample inner,
    const Sample outer,
    const float sharpness,
) {
  let alpha = compute_alpha(inner, outer, sharpness);
  let color = accumulator.rgb + inner.rgb * accumulator.transmittance * alpha;
  let transmittance = accumulator.transmittance * (1 - alpha);
  return { color, transmittance };
}

[Differentiable]
void store_accumulator(
    const Accumulator accumulator,
    const DiffTensorView out_accumulators,
    const uint camera_index,
    const uint2 pixel_coordinates,
) {
  let row = pixel_coordinates.y;
  let col = pixel_coordinates.x;

  // Save the RGB channels.
  store_once(out_accumulators, uint4(camera_index, 0, row, col), accumulator.rgb.r);
  store_once(out_accumulators, uint4(camera_index, 1, row, col), accumulator.rgb.g);
  store_once(out_accumulators, uint4(camera_index, 2, row, col), accumulator.rgb.b);

  // Save transmittance.
  store_once(
      out_accumulators,
      uint4(camera_index, 3, row, col),
      accumulator.transmittance,
  );
}

[Differentiable]
Accumulator load_accumulator(
    const DiffTensorView out_accumulators,
    const uint camera_index,
    const uint2 pixel_coordinates,
) {
  let row = pixel_coordinates.y;
  let col = pixel_coordinates.x;
  return {
    float3(
        load(out_accumulators, uint4(camera_index, 0, row, col)),
        load(out_accumulators, uint4(camera_index, 1, row, col)),
        load(out_accumulators, uint4(camera_index, 2, row, col)),
    ),
    load(out_accumulators, uint4(camera_index, 3, row, col)),
  };
}

[Differentiable]
float load_sharpness(const DiffTensorView sharpness) {
  return sharpness[0];
}

[AutoPyBindCUDA]
[CUDAKernel]
void intersect_and_composite_forward(
    const TensorView<int> sorted_triangle_indices,
    const TensorView<int64_t> tile_boundaries,
    const TensorView<float> vertices,
    const DiffTensorView colors,
    const DiffTensorView signed_distances,
    const TensorView<int> faces,
    const DiffTensorView sharpness,
    const int num_shells,
    const int grid_num_rows,
    const int grid_num_cols,
    const DiffTensorView out_accumulators,
    const TensorView<int64_t> sorted_keys,
) {
  let num_cameras = tile_boundaries.size(0);
  let image_height = out_accumulators.size(2);
  let image_width = out_accumulators.size(3);
  const uint num_tiles = grid_num_rows * grid_num_cols;

  // Each CUDA block corresponds to a single tile. The blocks use flat indexing, so we
  // have to compute the corresponding 4D index (camera, shell, row, column) for each
  // tile.
  let block_index = cudaBlockIdx().x;
  let tile_index = block_index % num_tiles;
  let tile_xy = uint2(tile_index % grid_num_cols, tile_index / grid_num_cols);
  let camera_index = block_index / num_tiles;

  // The indices within each block also use flat indexing, so we convert the flat index
  // to an XY index within the block. We then combine the index within the block with
  // the block's index to compute the XY pixel location for this thread.
  let index_in_block = cudaThreadIdx().x;
  let xy_in_tile = uint2(index_in_block % TILE_WIDTH, index_in_block / TILE_WIDTH);
  let xy = tile_xy * uint2(TILE_WIDTH, TILE_HEIGHT) + xy_in_tile;

  // Load the sharpness, which is needed to compute alpha values from signed distances.
  let loaded_sharpness = load_sharpness(sharpness);

  // Check whether the current thread's pixel is within the image.
  let is_in_image = (xy.x < image_width) && (xy.y < image_height);

  // Keep track of the pixel's RGB and transmittance, plus whether triangles should be
  // counted. Triangles are not counted if the current thread's pixel is not in the
  // image or if the transmittance becomes close to zero.
  Accumulator accumulator = { float3(0.f, 0.f, 0.f), 1.f };
  var is_active = is_in_image;

  // This counter indicates which shell index was last rendered. A triangle is only
  // rendered if its shell index is greater than all previously rendered triangles'
  // shell indices. If no triangle has been rendered yet, the outer sample is a dummy
  // sample with CDF == 0.
  int last_shell = -1;
  Sample outer_sample;

  // Triangles within the tile are processed in block-sized chunks. Calculate the number
  // of such chunks that are needed to process all triangles.
  let tile_start = load(tile_boundaries, uint64_t3(camera_index, tile_index, 0));
  let tile_end = load(tile_boundaries, uint64_t3(camera_index, tile_index, 1));
  let block_size = TILE_HEIGHT * TILE_WIDTH;
  let num_chunks = (tile_end - tile_start + block_size - 1) / block_size;
  var remaining_triangles = tile_end - tile_start;
  for (int chunk_index = 0; chunk_index < num_chunks; chunk_index++) {
    // Collectively fetch per-triangle data to shared memory.
    AllMemoryBarrierWithGroupSync();
    int64_t key_index = tile_start + chunk_index * block_size + index_in_block;
    if (key_index < tile_end) {
      let loaded_triangle_index = load(sorted_triangle_indices, key_index);
      let shell_index = extract_shell_index(load(sorted_keys, key_index));
      shared_triangles[index_in_block] = load_triangle(
          vertices,
          colors,
          signed_distances,
          faces,
          camera_index,
          shell_index,
          loaded_triangle_index,
      );
    }
    AllMemoryBarrierWithGroupSync();

    if (is_active) {
      for (int i = 0; i < min(block_size, remaining_triangles); i++) {
        let triangle = shared_triangles[i];

        // Evaluate the current triangle and skip it if it wasn't hit.
        let inner_sample = sample_triangle(triangle, xy + 0.5f, last_shell);
        if (!inner_sample.hit) {
          continue;
        }

        // For the first sample, use a dummy outer sample with a CDF of zero.
        accumulator = accumulate(
            accumulator,
            inner_sample,
            (last_shell == -1) ? Sample(float3(0, 0, 0), 1e10, true) : outer_sample,
            loaded_sharpness
        );

        // Update the bookkeeping values.
        outer_sample = inner_sample;
        last_shell = triangle.shell;
        if (last_shell == num_shells - 1) {
          is_active = false;
          break;
        }
      }

      remaining_triangles -= block_size;
    }
  }

  if (is_in_image) {
    store_accumulator(accumulator, out_accumulators, camera_index, xy);
  }
}

[AutoPyBindCUDA]
[CUDAKernel]
void intersect_and_composite_backward(
    const TensorView<int> sorted_triangle_indices,
    const TensorView<int64_t> tile_boundaries,
    const TensorView<float> vertices,
    const DiffTensorView colors,
    const DiffTensorView signed_distances,
    const TensorView<int> faces,
    const DiffTensorView sharpness,
    const int num_shells,
    const int grid_num_rows,
    const int grid_num_cols,
    const DiffTensorView out_accumulators,
    const TensorView<int64_t> sorted_keys,
) {
  let num_cameras = tile_boundaries.size(0);
  let image_height = out_accumulators.size(2);
  let image_width = out_accumulators.size(3);
  const uint num_tiles = grid_num_rows * grid_num_cols;

  // Each CUDA block corresponds to a single tile. The blocks use flat indexing, so we
  // have to compute the corresponding 4D index (camera, shell, row, column) for each
  // tile.
  let block_index = cudaBlockIdx().x;
  let tile_index = block_index % num_tiles;
  let tile_xy = uint2(tile_index % grid_num_cols, tile_index / grid_num_cols);
  let camera_index = block_index / num_tiles;

  // The indices within each block also use flat indexing, so we convert the flat
  // index to an XY index within the block. We then combine the index within the block
  // with the block's index to compute the XY pixel location for this thread.
  let index_in_block = cudaThreadIdx().x;
  let xy_in_tile = uint2(index_in_block % TILE_WIDTH, index_in_block / TILE_WIDTH);
  let xy = tile_xy * uint2(TILE_WIDTH, TILE_HEIGHT) + xy_in_tile;

  // Load the sharpness, which is needed to compute alpha values from signed
  // distances.
  let loaded_sharpness = load_sharpness(sharpness);

  // Check whether the current thread's pixel is within the image.
  let is_in_image = (xy.x < image_width) && (xy.y < image_height);

  // Keep track of the pixel's RGB and transmittance, plus whether triangles should be
  // counted. Triangles are not counted if the current thread's pixel is not in the
  // image or if the transmittance becomes close to zero.
  Accumulator accumulator = { float3(0.f, 0.f, 0.f), 1.f };

  // Reset the bookkeeping values for the second loop.
  int last_shell = -1;
  Sample outer_sample;
  var is_active = is_in_image;

  // Since each alpha value depends on two samples' SDF values, we need to accumulate
  // each sample's differential across two triangles/loop iterations.
  var d_sample_partial = Sample.Differential(float3(0, 0, 0), 0);
  var d_sample_partial_index = -1;

  // Load the accumulator and its gradients.
  Accumulator final_accumulator;
  if (is_in_image) {
    final_accumulator = load_accumulator(out_accumulators, camera_index, xy);
  }
  var dp_final_accumulator = diffPair(final_accumulator);
  if (is_in_image) {
    bwd_diff(store_accumulator)(
        dp_final_accumulator,
        out_accumulators,
        camera_index,
        xy,
    );
  }

  // Triangles within the tile are processed in block-sized chunks. Calculate the
  // number of such chunks that are needed to process all triangles.
  let tile_start = load(tile_boundaries, uint64_t3(camera_index, tile_index, 0));
  let tile_end = load(tile_boundaries, uint64_t3(camera_index, tile_index, 1));
  let block_size = TILE_HEIGHT * TILE_WIDTH;
  let num_chunks = (tile_end - tile_start + block_size - 1) / block_size;
  float d_sharpness = 0;
  Triangle last_triangle;
  var remaining_triangles = tile_end - tile_start;
  for (int chunk_index = 0; chunk_index < num_chunks; chunk_index++) {
    // Collectively fetch per-triangle data to shared memory.
    AllMemoryBarrierWithGroupSync();
    uint64_t key_index = tile_start + chunk_index * block_size + index_in_block;
    if (key_index < tile_end) {
      let loaded_triangle_index = load(sorted_triangle_indices, key_index);
      let shell_index = extract_shell_index(load(sorted_keys, key_index));
      shared_triangles[index_in_block] = load_triangle(
          vertices,
          colors,
          signed_distances,
          faces,
          camera_index,
          shell_index,
          loaded_triangle_index,
      );
      shared_triangle_indices[index_in_block] = loaded_triangle_index;
    }
    AllMemoryBarrierWithGroupSync();

    if (is_active) {
      for (int i = 0; i < min(block_size, remaining_triangles); i++) {
        let triangle = shared_triangles[i];

        // Evaluate the current triangle and skip it if it wasn't hit.
        let inner_sample = sample_triangle(triangle, xy + 0.5f, last_shell);
        if (!inner_sample.hit) {
          continue;
        }
        last_triangle = triangle;

        // For the first sample, use a dummy outer sample with a CDF of zero.
        let dummy = Sample(float3(0, 0, 0), 1e10, true);
        let outer_sample_or_dummy = (last_shell == -1) ? dummy : outer_sample;
        let alpha =
            compute_alpha(inner_sample, outer_sample_or_dummy, loaded_sharpness);

        // Use the classic suffix trick to compute gradients for this layer's color
        // and alpha values. Note that d_color applies to the inner sample.
        let weight = alpha * accumulator.transmittance;
        accumulator.rgb += weight * inner_sample.rgb;
        let suffix = final_accumulator.rgb - accumulator.rgb;
        let inverse_alpha = 1 / (1 - alpha + EPSILON);
        let d_color = weight * dp_final_accumulator.d.rgb;
        let d_alpha_from_color =
            dot(dp_final_accumulator.d.rgb,
                accumulator.transmittance * inner_sample.rgb - suffix * inverse_alpha);
        let d_alpha_from_transmittance = -dp_final_accumulator.d.transmittance *
                                         final_accumulator.transmittance *
                                         inverse_alpha;
        let d_alpha = d_alpha_from_color + d_alpha_from_transmittance;

        // Backpropagate through computing the alpha value.
        var dp_inner_sample = diffPair(inner_sample);
        var dp_outer_sample = diffPair(outer_sample_or_dummy);
        var dp_sharpness = diffPair(loaded_sharpness);
        bwd_diff(compute_alpha)(
            dp_inner_sample,
            dp_outer_sample,
            dp_sharpness,
            d_alpha
        );
        d_sharpness += dp_sharpness.d;

        // For all triangles except the first triangle, accumulate the resulting
        // gradients in d_sample_partial, then backpropagate through sampling and
        // loading the triangle. d_sample_partial already contains the gradients from
        // when the triangle corresponded to the inner sample; we add the gradients
        // from when it corresponds to the outer sample.
        if (last_shell != -1) {
          // Note that the outer sample does not contribute to the color.
          d_sample_partial.sdf += dp_outer_sample.d.sdf;

          // Backpropagate through loading and evaluating the triangle.
          var dp_triangle = diffPair(triangle);
          bwd_diff(sample_triangle)(
              dp_triangle,
              xy + 0.5f,
              last_shell,
              d_sample_partial,
          );
          bwd_diff(load_triangle)(
              vertices,
              colors,
              signed_distances,
              faces,
              camera_index,
              0, // The shell index isn't used during backprop, so use a dummy value.
              d_sample_partial_index,
              dp_triangle.d,
          );
        }

        // The next d_sample_partial becomes dSample from the inner sample. Note that
        // backpropagating through compute_alpha does not send gradients to the sample
        // colors; the sample color gradients come directly from compositing.
        d_sample_partial = Sample.Differential(d_color, dp_inner_sample.d.sdf);
        d_sample_partial_index = shared_triangle_indices[i];

        // This has to be updated after computing the gradients.
        accumulator.transmittance *= 1 - alpha;

        // Update the bookkeeping values.
        outer_sample = inner_sample;
        last_shell = triangle.shell;
        if (last_shell == num_shells - 1) {
          is_active = false;
          break;
        }
      }

      remaining_triangles -= block_size;
    }
  }

  // Everything after this point is only relevant if the thread corresponds to a pixel
  // that's in the image.
  if (!is_in_image) {
    return;
  }

  if (last_shell != -1) {
    // Decrement last_shell so that sample_triangle doesn't exit early.
    last_shell--;

    // Backpropagate through loading and evaluating the last triangle.
    var dp_triangle = diffPair(last_triangle);
    bwd_diff(sample_triangle)(dp_triangle, xy + 0.5f, last_shell, d_sample_partial, );
    bwd_diff(load_triangle)(
        vertices,
        colors,
        signed_distances,
        faces,
        camera_index,
        0, // The shell index isn't used during backprop, so use a dummy value.
        d_sample_partial_index,
        dp_triangle.d,
    );
  }

  bwd_diff(load_sharpness)(sharpness, d_sharpness);
}
