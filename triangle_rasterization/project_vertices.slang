import "./common.slang";

struct Camera {
  float4x4 extrinsics;
  float3x3 intrinsics;
}

Camera load_camera(
    const TensorView<float> extrinsics,
    const TensorView<float> intrinsics,
    const uint camera_index,
) {
  let w2c = float4x4(
      extrinsics[camera_index, 0, 0],
      extrinsics[camera_index, 0, 1],
      extrinsics[camera_index, 0, 2],
      extrinsics[camera_index, 0, 3],
      extrinsics[camera_index, 1, 0],
      extrinsics[camera_index, 1, 1],
      extrinsics[camera_index, 1, 2],
      extrinsics[camera_index, 1, 3],
      extrinsics[camera_index, 2, 0],
      extrinsics[camera_index, 2, 1],
      extrinsics[camera_index, 2, 2],
      extrinsics[camera_index, 2, 3],
      extrinsics[camera_index, 3, 0],
      extrinsics[camera_index, 3, 1],
      extrinsics[camera_index, 3, 2],
      extrinsics[camera_index, 3, 3],
  );
  let k = float3x3(
      intrinsics[camera_index, 0, 0],
      intrinsics[camera_index, 0, 1],
      intrinsics[camera_index, 0, 2],
      intrinsics[camera_index, 1, 0],
      intrinsics[camera_index, 1, 1],
      intrinsics[camera_index, 1, 2],
      intrinsics[camera_index, 2, 0],
      intrinsics[camera_index, 2, 1],
      intrinsics[camera_index, 2, 2],
  );

  return { w2c, k };
}

struct Vertex2D {
  float2 xy;
  float depth;
}

Vertex2D project_vertex(const float3 vertex, const Camera camera) {
  // Transform the vertex into camera space.
  let cameraSpaceVertex = mul(camera.extrinsics, float4(vertex, 1));

  // Apply the intrinsics and divide by depth.
  let depth = cameraSpaceVertex.z;
  let xy = mul(camera.intrinsics, cameraSpaceVertex.xyz).xy / depth;
  return { xy, depth };
}

[AutoPyBindCUDA]
[CUDAKernel]
void project_vertices(
    const TensorView<float> extrinsics,
    const TensorView<float> intrinsics,
    const TensorView<float> vertices,
    const TensorView<float> out_vertices,
    const TensorView<float> out_depths,
) {
  // Skip past invalid vertices.
  let vertex_index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (vertex_index >= vertices.size(0)) {
    return;
  }

  // Load the vertex's position.
  let vertex = float3(
      vertices[vertex_index, 0],
      vertices[vertex_index, 1],
      vertices[vertex_index, 2],
  );

  // Loop through the cameras and project the vertex to each one.
  let num_cameras = extrinsics.size(0);
  for (uint camera_index = 0; camera_index < num_cameras; camera_index++) {
    let camera = load_camera(extrinsics, intrinsics, camera_index);
    let projected = project_vertex(vertex, camera);
    store(out_vertices, uint3(camera_index, vertex_index, 0), projected.xy.x);
    store(out_vertices, uint3(camera_index, vertex_index, 1), projected.xy.y);
    store(out_depths, uint2(camera_index, vertex_index), projected.depth);
  }
}
