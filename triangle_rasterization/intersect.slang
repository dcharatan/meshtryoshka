import "./common.slang";

struct Face {
  int index;
  float2 a;
  float2 b;
  float2 c;
};

// Define the shared memory that each CUDA block uses to load triangles.
static const uint NUM_SHARED_FACES = TILE_HEIGHT * TILE_WIDTH;
groupshared Face shared_faces[NUM_SHARED_FACES];

float3 compute_barycentric_coordinates(const Face face, const float2 point) {
  // Compute dot products.
  let d_12_12 = dot(face.b - face.a, face.b - face.a);
  let d_12_23 = dot(face.b - face.a, face.c - face.b);
  let d_12_13 = dot(face.b - face.a, face.c - face.a);
  let d_13_13 = dot(face.c - face.a, face.c - face.a);
  let d_13_23 = dot(face.c - face.a, face.c - face.b);
  let d_23_23 = dot(face.c - face.b, face.c - face.b);
  let d_1x_12 = dot(point - face.a, face.b - face.a);
  let d_1x_13 = dot(point - face.a, face.c - face.a);
  let d_2x_12 = dot(point - face.b, face.b - face.a);
  let d_2x_13 = dot(point - face.b, face.c - face.a);

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
struct Sample {
  float3 uvw; // barycentric coordinates
  bool hit;   // whether the triangle was intersected
}

Sample sample_triangle(const Face face, const float2 point) {
  // Compute barycentric coordinates for the triangle.
  let uvw = compute_barycentric_coordinates(face, point);

  // If the barycentric coordinates land outside the triangle or appear to be
  // degenerate, return no hit.
  let sum = uvw.x + uvw.y + uvw.z;
  let miss = isnan(uvw.x) || isnan(uvw.y) || isnan(uvw.z) || uvw.x < 0 || uvw.x > 1 ||
             uvw.y < 0 || uvw.y > 1 || uvw.z < 0 || uvw.z > 1 || sum < 0.999 ||
             sum > 1.001;

  return { uvw, !miss };
}

float2 load_vertex(
    const TensorView<float> vertices,
    const TensorView<int> faces,
    const int camera_index,
    const int face_index,
    const int vertex_index_in_face,
) {
  let vertex_index = load(faces, uint64_t2(face_index, vertex_index_in_face));
  return {
    load(vertices, uint64_t3(camera_index, vertex_index, 0)),
    load(vertices, uint64_t3(camera_index, vertex_index, 1)),
  };
}

[AutoPyBindCUDA]
[CUDAKernel]
void intersect(
    const TensorView<int> sorted_triangle_indices,
    const TensorView<int64_t> tile_boundaries,
    const TensorView<float> vertices,
    const TensorView<int> faces,
    const int grid_num_rows,
    const int grid_num_cols,
    const int image_height,
    const int image_width,
    const TensorView<float> out_uv,
    const TensorView<int> out_index,
) {
  let num_cameras = tile_boundaries.size(0);
  let num_shells = tile_boundaries.size(1);
  const uint num_tiles = grid_num_rows * grid_num_cols;

  // Each CUDA block corresponds to a single tile. The blocks use flat indexing, so we
  // have to compute the corresponding 4D index (camera, shell, row, column) for each
  // tile.
  let block_index = cudaBlockIdx().x;
  let tile_index = block_index % num_tiles;
  let tile_xy = uint2(tile_index % grid_num_cols, tile_index / grid_num_cols);
  let batch_index = block_index / num_tiles;
  let shell_index = batch_index % num_shells;
  let camera_index = batch_index / num_shells;

  // The indices within each block also use flat indexing, so we convert the flat index
  // to an XY index within the block. We then combine the index within the block with
  // the block's index to compute the XY pixel location for this thread.
  let index_in_block = cudaThreadIdx().x;
  let xy_in_tile = uint2(index_in_block % TILE_WIDTH, index_in_block / TILE_WIDTH);
  let xy = tile_xy * uint2(TILE_WIDTH, TILE_HEIGHT) + xy_in_tile;

  // Check whether the current thread's pixel is within the image.
  let is_in_image = (xy.x < image_width) && (xy.y < image_height);

  // Triangles within the tile are processed in block-sized chunks. Calculate the number
  // of such chunks that are needed to process all triangles.
  var is_active = is_in_image;
  let tile_start =
      load(tile_boundaries, uint64_t4(camera_index, shell_index, tile_index, 0));
  let tile_end =
      load(tile_boundaries, uint64_t4(camera_index, shell_index, tile_index, 1));
  let block_size = cudaBlockDim().x;
  let num_chunks = (tile_end - tile_start + block_size - 1) / block_size;
  var remaining_triangles = tile_end - tile_start;
  for (int chunkIndex = 0; chunkIndex < num_chunks; chunkIndex++) {
    // Collectively fetch per-triangle data to shared memory.
    AllMemoryBarrierWithGroupSync();
    let key_index = tile_start + chunkIndex * block_size + index_in_block;
    if (key_index < tile_end) {
      // Decompose the 64-bit key index into two 32-bit indices and use it to load the
      // triangle index using a special function. This is necessary because Slang uses
      // 32-bit math for array indexing, and the key array contains more than 2^32
      // bytes.
      let face_index = load(sorted_triangle_indices, key_index);
      shared_faces[index_in_block] = {
        face_index,
        load_vertex(vertices, faces, camera_index, face_index, 0),
        load_vertex(vertices, faces, camera_index, face_index, 1),
        load_vertex(vertices, faces, camera_index, face_index, 2),
      };
    }
    AllMemoryBarrierWithGroupSync();

    if (is_active) {
      for (int i = 0; i < min(block_size, remaining_triangles); i++) {
        let face = shared_faces[i];

        // Evaluate the current triangle and skip it if it wasn't hit.
        let sample = sample_triangle(face, xy + 0.5f);
        if (!sample.hit) {
          continue;
        }

        // Handle a triangle hit.
        is_active = false;
        if (is_in_image) {
          store(out_uv, uint64_t4(batch_index, xy.y, xy.x, 0), sample.uvw.x);
          store(out_uv, uint64_t4(batch_index, xy.y, xy.x, 1), sample.uvw.y);
          store(out_index, uint64_t3(batch_index, xy.y, xy.x), face.index);
        }
        break;
      }

      remaining_triangles -= block_size;
    }
  }

  // Handle the case where no triangles were hit.
  if (is_active) {
    store(out_uv, uint64_t4(batch_index, xy.y, xy.x, 0), 0.f);
    store(out_uv, uint64_t4(batch_index, xy.y, xy.x, 1), 0.f);
    store(out_index, uint64_t3(batch_index, xy.y, xy.x), -1);
  }
}
