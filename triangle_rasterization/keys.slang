static const uint64_t CAMERA_MASK = ((1 << CAMERA_BIT_LENGTH) - 1);
static const uint64_t SHELL_MASK = ((1 << SHELL_BIT_LENGTH) - 1);
static const uint64_t TILE_MASK = ((1 << TILE_BIT_LENGTH) - 1);
static const uint64_t DEPTH_MASK = 0xFFFFFFFF;

uint64_t pack_key(
    const uint camera_index,
    const uint shell_index,
    const uint tile_index,
    const float depth,
) {
  const uint64_t camera_bits = (camera_index & CAMERA_MASK) << CAMERA_BIT_OFFSET;
  const uint64_t shell_bits = (shell_index & SHELL_MASK) << SHELL_BIT_OFFSET;
  const uint64_t tile_bits = (tile_index & TILE_MASK) << TILE_BIT_OFFSET;

  // There are always 32 depth bits, so no mask is necessary.
  let depth_bits = ((uint64_t)reinterpret<uint>(depth)) << DEPTH_BIT_OFFSET;

  // This is an arbitrary ordering
  // The actual ordering coems from the BIT_OFFSET values
  return camera_bits | shell_bits | tile_bits | depth_bits;
}

uint extract_camera_index(const uint64_t key) {
  return (uint)((key >> CAMERA_BIT_OFFSET) & CAMERA_MASK);
}

uint extract_shell_index(const uint64_t key) {
  return (uint)((key >> SHELL_BIT_OFFSET) & SHELL_MASK);
}

uint extract_tile_index(const uint64_t key) {
  return (uint)((key >> TILE_BIT_OFFSET) & TILE_MASK);
}
