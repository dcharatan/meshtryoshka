int logical_right_shift(const int x, const int shift) {
  // In Slang, int uses arithmetic right shift, and uint uses logical right shift. Thus,
  // to get a logical right shift for int, we have to cast to uint and back.
  return reinterpret<int>(reinterpret<uint>(x) >> shift);
}

[AutoPyBindCUDA]
[CUDAKernel]
void dilate_occupancy(
    const TensorView<int> occupancy,
    const TensorView<int> new_occupancy,
    const int dilation,
) {
  // Read the packed occupancy grid's dimensions.
  let i_max = occupancy.size(0);
  let j_max = occupancy.size(1);
  let k_packed_max = occupancy.size(2);

  // Guard against indices that are invalid in the occupancy grid.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let k_packed = index.z;
  if (i >= i_max || j >= j_max || k_packed >= k_packed_max) {
    return;
  }

  int result = 0;

  // Handle the previous k index.
  if (k_packed > 0) {
    int at_previous = 0;
    for (int di = -dilation; di <= dilation; di++) {
      for (int dj = -dilation; dj <= dilation; dj++) {
        let ii = i + di;
        let jj = j + dj;
        if (ii < 0 || ii >= i_max || jj < 0 || jj >= j_max) {
          continue;
        }
        at_previous |= occupancy[ii, jj, k_packed - 1];
      }
    }
    at_previous = logical_right_shift(at_previous, 32 - dilation);
    for (int bit = 0; bit < dilation; bit++) {
      result |= at_previous;
      at_previous = logical_right_shift(at_previous, 1);
    }
  }

  // Handle the current k index.
  int at_current = 0;
  for (int di = -dilation; di <= dilation; di++) {
    for (int dj = -dilation; dj <= dilation; dj++) {
      let ii = i + di;
      let jj = j + dj;
      if (ii < 0 || ii >= i_max || jj < 0 || jj >= j_max) {
        continue;
      }
      at_current |= occupancy[ii, jj, k_packed];
    }
  }
  result |= at_current;
  int at_current_left = at_current;
  int at_current_right = at_current;
  for (int bit = 0; bit < dilation; bit++) {
    at_current_right = logical_right_shift(at_current_right, 1);
    at_current_left <<= 1;
    result |= at_current_left;
    result |= at_current_right;
  }

  // Handle the next k index.
  if (k_packed < k_packed_max - 1) {
    int at_next = 0;
    for (int di = -dilation; di <= dilation; di++) {
      for (int dj = -dilation; dj <= dilation; dj++) {
        let ii = i + di;
        let jj = j + dj;
        if (ii < 0 || ii >= i_max || jj < 0 || jj >= j_max) {
          continue;
        }
        at_next |= occupancy[ii, jj, k_packed + 1];
      }
    }
    at_next <<= (32 - dilation);
    for (int bit = 0; bit < dilation; bit++) {
      result |= at_next;
      at_next <<= 1;
    }
  }

  // Write the result to the new occupancy grid.
  new_occupancy[i, j, k_packed] = result;
}
