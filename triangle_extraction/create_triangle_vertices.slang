import "./common.slang";

struct Voxel : IDifferentiable {
  // The origin (always valid).
  float d0;
  float3 v0;
  float sh0[NUM_SPHERICAL_HARMONICS][3];

  // The axes (sometimes invalid).
  float d1[3];
  float3 v1[3];
  float sh1[3][NUM_SPHERICAL_HARMONICS][3];
}

[Differentiable]
Voxel load_voxel(
    const int index,
    const DiffTensorView grid_vertices,
    const DiffTensorView grid_signed_distances,
    const DiffTensorView grid_spherical_harmonics,
    const TensorView<int> voxel_lower_corners,
) {
  // Load the origin's signed distance (d0), vertex location (v0), and features (f0).
  let i0 = voxel_lower_corners[0, index];
  let d0 = grid_signed_distances[i0];
  let v0 = float3(grid_vertices[i0, 0], grid_vertices[i0, 1], grid_vertices[i0, 2]);
  float sh0[NUM_SPHERICAL_HARMONICS][3];
  $for(s in Range(0, NUM_SPHERICAL_HARMONICS)) {
    $for(j in Range(0, 3)) {
      sh0[s][j] = load(grid_spherical_harmonics, uint3(s, i0, j));
    }
  }

  // Load the signed distance (d1), vertex location (v1), and features (f1) for the
  // +X, +Y, and +Z directions.
  float d1[3];
  float3 v1[3];
  float sh1[3][NUM_SPHERICAL_HARMONICS][3];
  for (int axis = 0; axis < 3; axis++) {
    if (voxel_lower_corners[axis + 1, index] >= 0) {
      // Handle the case where the axis is valid.
      let i1 = voxel_lower_corners[axis + 1, index];
      d1[axis] = grid_signed_distances[i1];
      v1[axis] =
          float3(grid_vertices[i1, 0], grid_vertices[i1, 1], grid_vertices[i1, 2]);
      $for(s in Range(0, NUM_SPHERICAL_HARMONICS)) {
        $for(j in Range(0, 3)) {
          sh1[axis][s][j] = load(grid_spherical_harmonics, uint3(s, i1, j));
        }
      }
    } else {
      // Handle the case where the axis is invalid. Simply setting d1[axis] to d0
      // prevents a vertex from being placed.
      d1[axis] = d0;
    }
  }

  return { d0, v0, sh0, d1, v1, sh1 };
}

bool should_compute_triangle_vertex(
    const Voxel voxel,
    const int axis,
    no_diff const float level_set,
) {
  // Vertices always lie along voxel edges. A vertex is created wherever an edge's
  // endpoints' signed distances are on opposite sides of the level set.
  if (voxel.d0 < level_set && voxel.d1[axis] < level_set) {
    return false;
  }
  if (voxel.d0 >= level_set && voxel.d1[axis] >= level_set) {
    return false;
  }
  return true;
}

struct TriangleVertex : IDifferentiable {
  float3 v;
  float sh[NUM_SPHERICAL_HARMONICS][3];
  float d;
  uint8_t vertex_type;
}

[Differentiable]
TriangleVertex compute_triangle_vertex(
    const Voxel voxel,
    const int axis,
    no_diff const float level_set,
) {
  // Compute the interpolation weight along the edge.
  // Detach removes gradients from voxel sdf -> mesh vertex
  let t = detach(
      (voxel.d0 == voxel.d1[axis])
          ? 0.5
          : clamp((level_set - voxel.d0) / (voxel.d1[axis] - voxel.d0), 0, 1)
  );

  // Compute and the vertex location.
  // Gradients from voxel vertex -> mesh vertex.
  // The t here gives gradients from voxel sdf -> mesh vertex (but is detached)
  let v = voxel.v0 + t * (voxel.v1[axis] - voxel.v0);

  // Compute the feature values.
  float sh[NUM_SPHERICAL_HARMONICS][3];
  $for(s in Range(0, NUM_SPHERICAL_HARMONICS)) {
    $for(j in Range(0, 3)) {
      sh[s][j] = voxel.sh0[s][j] + float(t) * (voxel.sh1[axis][s][j] - voxel.sh0[s][j]);
    }
  }

  // Compute the vertex's signed distance.
  // This gives gradients from voxel sdf -> output sdf.
  // The t here is another path from voxel sdf -> output sdf (but is detached)
  // but this path is a more complex function of the voxel sdf.
  let d = voxel.d0 + t * (voxel.d1[axis] - voxel.d0);

  return { v, sh, d, uint8_t(axis) };
}

[Differentiable]
void store_triangle_vertex(
    const int index,
    const TriangleVertex triangle_vertex,
    const DiffTensorView triangle_vertices,
    const DiffTensorView triangle_spherical_harmonics,
    const DiffTensorView triangle_signed_distances,
    const TensorView<uint8_t> triangle_vertex_types,
) {
  triangle_vertices.storeOnce(uint2(index, 0), triangle_vertex.v.x);
  triangle_vertices.storeOnce(uint2(index, 1), triangle_vertex.v.y);
  triangle_vertices.storeOnce(uint2(index, 2), triangle_vertex.v.z);
  $for(s in Range(0, NUM_SPHERICAL_HARMONICS)) {
    $for(j in Range(0, 3)) {
      store_once(
          triangle_spherical_harmonics,
          uint3(s, index, j),
          triangle_vertex.sh[s][j]
      );
    }
  }
  triangle_signed_distances.storeOnce(index, triangle_vertex.d);
  triangle_vertex_types[index] = triangle_vertex.vertex_type;
}

[AutoPyBindCUDA]
[CUDAKernel]
void create_triangle_vertices_forward(
    const DiffTensorView grid_vertices,
    const DiffTensorView grid_signed_distances,
    const DiffTensorView grid_spherical_harmonics,
    const TensorView<int> voxel_lower_corners,
    const TensorView<int> vertex_offsets,
    const TensorView<float> level_sets,
    const DiffTensorView triangle_vertices,
    const DiffTensorView triangle_signed_distances,
    const DiffTensorView triangle_spherical_harmonics,
    const TensorView<uint8_t> triangle_vertex_types,
) {
  // Skip out-of-bounds voxels.
  int voxel_index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (voxel_index >= voxel_lower_corners.size(1)) {
    return;
  }

  // Load relevant grid information for the current voxel.
  let voxel = load_voxel(
      voxel_index,
      grid_vertices,
      grid_signed_distances,
      grid_spherical_harmonics,
      voxel_lower_corners,
  );

  // Loop through level sets and axes. At most, there's one vertex per axis.
  for (int level_set_idx = 0; level_set_idx < level_sets.size(0); level_set_idx++) {
    var vertex_index = vertex_offsets[level_set_idx, voxel_index];
    let level_set = level_sets[level_set_idx];
    for (int axis = 0; axis < 3; axis++) {
      // Skip past invalid vertices.
      if (!should_compute_triangle_vertex(voxel, axis, level_set)) {
        continue;
      }

      // Compute and store the triangle vertex.
      let triangle_vertex = compute_triangle_vertex(voxel, axis, level_set);
      store_triangle_vertex(
          vertex_index,
          triangle_vertex,
          triangle_vertices,
          triangle_spherical_harmonics,
          triangle_signed_distances,
          triangle_vertex_types,
      );

      // Update the vertex index outside the differentiable functions.
      vertex_index++;
    }
  }
}

[AutoPyBindCUDA]
[CUDAKernel]
void create_triangle_vertices_backward(
    const DiffTensorView grid_vertices,
    const DiffTensorView grid_signed_distances,
    const DiffTensorView grid_spherical_harmonics,
    const TensorView<int> voxel_lower_corners,
    const TensorView<int> vertex_offsets,
    const TensorView<float> level_sets,
    const DiffTensorView triangle_vertices,
    const DiffTensorView triangle_signed_distances,
    const DiffTensorView triangle_spherical_harmonics,
    const TensorView<uint8_t> triangle_vertex_types,
) {
  // Skip out-of-bounds voxels.
  int voxel_index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (voxel_index >= voxel_lower_corners.size(1)) {
    return;
  }

  // Load relevant grid information for the current voxel.
  let voxel = load_voxel(
      voxel_index,
      grid_vertices,
      grid_signed_distances,
      grid_spherical_harmonics,
      voxel_lower_corners,
  );

  // Initialize an empty differential.
  Voxel.Differential d_voxel = diffPair(voxel).d;

  // Loop through level sets and axes. At most, there's one vertex per axis.
  for (int level_set_idx = 0; level_set_idx < level_sets.size(0); level_set_idx++) {
    var vertex_index = vertex_offsets[level_set_idx, voxel_index];
    let level_set = level_sets[level_set_idx];
    for (int axis = 0; axis < 3; axis++) {
      // Skip past invalid vertices.
      if (!should_compute_triangle_vertex(voxel, axis, level_set)) {
        continue;
      }

      // Backpropagate through computing and storing the triangle vertex.
      let triangle_vertex = compute_triangle_vertex(voxel, axis, level_set);
      var dp_triangle_vertex = diffPair(triangle_vertex);
      bwd_diff(store_triangle_vertex)(
          vertex_index,
          dp_triangle_vertex,
          triangle_vertices,
          triangle_spherical_harmonics,
          triangle_signed_distances,
          triangle_vertex_types,
      );

      var dp_voxel = diffPair(voxel);
      bwd_diff(compute_triangle_vertex)(
          dp_voxel,
          axis,
          level_set,
          dp_triangle_vertex.d,
      );

      // Accumulate the voxel differential. Defining dp_voxel outside the loops and
      // passing it to bwd_diff(compute_triangle_vertex) would cause it to be
      // overwritten, which is why we need to do this.
      d_voxel.d0 += dp_voxel.d.d0;
      d_voxel.d1[axis] += dp_voxel.d.d1[axis];
      d_voxel.v0 += dp_voxel.d.v0;
      d_voxel.v1[axis] += dp_voxel.d.v1[axis];
      $for(s in Range(0, NUM_SPHERICAL_HARMONICS)) {
        $for(j in Range(0, 3)) {
          d_voxel.sh0[s][j] += dp_voxel.d.sh0[s][j];
          d_voxel.sh1[axis][s][j] += dp_voxel.d.sh1[axis][s][j];
        }
      }

      // Update the vertex index outside the differentiable functions.
      vertex_index++;
    }
  }

  // Backpropagate through loading the voxel.
  bwd_diff(load_voxel)(
      voxel_index,
      grid_vertices,
      grid_signed_distances,
      grid_spherical_harmonics,
      voxel_lower_corners,
      d_voxel,
  );
}
