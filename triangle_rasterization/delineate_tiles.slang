import "./common.slang";
import "./keys.slang";

uint64_t3 get_index(const uint64_t key, const bool separate_shells) {
  return {
    extract_camera_index(key),
    separate_shells ? extract_shell_index(key) : 0,
    extract_tile_index(key),
  };
}

[AutoPyBindCUDA]
[CUDAKernel]
void delineate_tiles(
    const TensorView<int64_t> sorted_keys,
    const TensorView<int64_t> num_keys_container,
    const TensorView<int64_t> out_tile_boundaries,
    const bool separate_shells,
    const float epsilon,
) {
  // Array dimensions in Slang are 32-bit. However, the number of keys may be larger
  // than 2^32. Thus, we pass the number of keys as an explicit 64-bit integer and
  // implement our own pointer arithmetic using the load function. Slang also
  // doesn't know how to bind 64-bit integers, so we pass the number of keys via a
  // TensorView with a single element.
  let num_keys = num_keys_container[0];
  let key_index = (int64_t)cudaBlockIdx().x * (int64_t)cudaBlockDim().x +
                  (int64_t)cudaThreadIdx().x;
  if (key_index >= num_keys) {
    return;
  }

  // Read the tile index from the key.
  let this_key = reinterpret<uint64_t>(load(sorted_keys, key_index));
  let this_index = get_index(this_key, separate_shells);

  // We assume that out_tile_boundaries has been zero-initialized. Thus, we don't need
  // to special-case the first tile's start index (i.e., set it to zero). Note that we
  // would still have to zero-initialize out_tile_boundaries even if we special-cased
  // the first tile's start index here, since not all tiles are guaranteed to be present
  // in the sorted keys.

  // For subsequent keys, detect the boundaries between tile indices.
  if (key_index > 0) {
    let last_key = reinterpret<uint64_t>(load(sorted_keys, key_index - 1));
    let last_index = get_index(last_key, separate_shells);

    if (any(this_index != last_index)) {
      store(out_tile_boundaries, uint64_t4(last_index, 1), key_index);
      store(out_tile_boundaries, uint64_t4(this_index, 0), key_index);
    }
  }

  // Special-case the end of the last tile.
  if (key_index == num_keys - 1) {
    store(out_tile_boundaries, uint64_t4(this_index, 1), num_keys);
  }
}
