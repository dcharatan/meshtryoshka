import "./common.slang";

Vertex load_vertex(
    const TensorView<float> vertices,
    const TensorView<float> depths,
    const int camera_index,
    const int vertex_index,
) {
  return {
    load(vertices, uint64_t3(camera_index, vertex_index, 0)),
    load(vertices, uint64_t3(camera_index, vertex_index, 1)),
    load(depths, uint64_t2(camera_index, vertex_index)),
  };
}

struct Vertex {
  float x;
  float y;
  float depth;
}

struct Triangle {
  Vertex a;
  Vertex b;
  Vertex c;
}

Triangle load_triangle(
    const TensorView<float> vertices,
    const TensorView<float> depths,
    const TensorView<int> faces,
    const int camera_index,
    const int face_index,
) {
  return {
    load_vertex(
        vertices,
        depths,
        camera_index,
        load(faces, uint64_t2(face_index, 0)),
    ),
    load_vertex(
        vertices,
        depths,
        camera_index,
        load(faces, uint64_t2(face_index, 1)),
    ),
    load_vertex(
        vertices,
        depths,
        camera_index,
        load(faces, uint64_t2(face_index, 2)),
    ),
  };
}

struct TileMembership {
  int2 y;
  int2 x;
}

int2 compute_tile_bounds(
    const float minimum,
    const float maximum,
    const int tile_size,
    const int min_bound,
    const int max_bound,
) {
  // Compute the minimum (inclusive) and maximum (exclusive) tile bounds for the given
  // minimum and maximum float values.
  let tile_minimum = int(minimum) / tile_size;
  let tile_maximum = int(ceil(maximum / tile_size));

  // Ensure that the tile bounds are valid by clamping them.
  return int2(
      clamp(tile_minimum, min_bound, max_bound),
      clamp(tile_maximum, min_bound, max_bound),
  );
}

TileMembership compute_tile_membership(
    const Triangle triangle,
    const int grid_tile_height,
    const int grid_tile_width,
    const int grid_row_minimum,
    const int grid_col_minimum,
    const int grid_row_maximum,
    const int grid_col_maximum,
    const float near_plane,
) {
  // Cull back-facing triangles by assigning them to no tiles.
  let a = triangle.a;
  let b = triangle.b;
  let c = triangle.c;
  let check = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  if (check < 0) {
    return TileMembership(int2(0, 0), int2(0, 0));
  }

  // Cull triangles in front of the near plane by assigning them to no tiles.
  let min_depth = min(min(a.depth, b.depth), c.depth);
  if (min_depth < near_plane) {
    return TileMembership(int2(0, 0), int2(0, 0));
  }

  // Compute tile bounds along each axis.
  return {
    compute_tile_bounds(
        min(min(a.y, b.y), c.y),
        max(max(a.y, b.y), c.y),
        grid_tile_height,
        grid_row_minimum,
        grid_row_maximum,
    ),
    compute_tile_bounds(
        min(min(a.x, b.x), c.x),
        max(max(a.x, b.x), c.x),
        grid_tile_width,
        grid_col_minimum,
        grid_col_maximum,
    ),
  };
}

void store_tile_membership(
    const TileMembership tile_membership,
    TensorView<int> out_tile_minima,
    TensorView<int> out_tile_maxima,
    TensorView<int> out_num_tiles_touched,
    const int camera_index,
    const int face_index,
) {
  store(out_tile_minima, uint64_t3(camera_index, face_index, 0), tile_membership.y[0]);
  store(out_tile_minima, uint64_t3(camera_index, face_index, 1), tile_membership.x[0]);

  store(out_tile_maxima, uint64_t3(camera_index, face_index, 0), tile_membership.y[1]);
  store(out_tile_maxima, uint64_t3(camera_index, face_index, 1), tile_membership.x[1]);

  let product = (tile_membership.y[1] - tile_membership.y[0]) *
                (tile_membership.x[1] - tile_membership.x[0]);
  store(out_num_tiles_touched, uint64_t2(camera_index, face_index), product);
}

[AutoPyBindCUDA]
[CUDAKernel]
void bound_triangles(
    const TensorView<float> vertices,
    const TensorView<float> depths,
    const TensorView<int> faces,
    const int grid_tile_height,
    const int grid_tile_width,
    const int grid_row_minimum,
    const int grid_col_minimum,
    const int grid_row_maximum,
    const int grid_col_maximum,
    const float near_plane,
    const TensorView<int> out_tile_minima,
    const TensorView<int> out_tile_maxima,
    const TensorView<int> out_num_tiles_touched,
) {
  // Skip past invalid triangles.
  let face_index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (face_index >= faces.size(0)) {
    return;
  }

  // Loop through the cameras.
  let num_cameras = vertices.size(0);
  for (int camera_index = 0; camera_index < num_cameras; camera_index++) {
    // Compute tile overlaps for the projected triangle.
    let tile_membership = compute_tile_membership(
        load_triangle(vertices, depths, faces, camera_index, face_index),
        grid_tile_height,
        grid_tile_width,
        grid_row_minimum,
        grid_col_minimum,
        grid_row_maximum,
        grid_col_maximum,
        near_plane,
    );

    // Store the results.
    store_tile_membership(
        tile_membership,
        out_tile_minima,
        out_tile_maxima,
        out_num_tiles_touched,
        camera_index,
        face_index,
    );
  }
}
