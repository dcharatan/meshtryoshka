///////////////////////////////////////
// 64-bit Indexing for TensorView<T> //
///////////////////////////////////////

uint64_t compute_offset<T, int N>(TensorView<T> x, vector<uint64_t, N> index) {
  uint64_t offset = 0;
  uint64_t stride = sizeof(T);
  for (int i = N - 1; i >= 1; i--) {
    offset += index[i] * stride;
    stride *= (uint64_t)x.size(i);
  }
  offset += index[0] * stride;
  return offset;
}

T load<T, int N>(TensorView<T> x, vector<uint64_t, N> index) {
  uint64_t offset = compute_offset(x, index);
  return *((T *)(((uint8_t *)x.data_ptr()) + offset));
}

T store<T, int N>(TensorView<T> x, vector<uint64_t, N> index, T value) {
  uint64_t offset = compute_offset(x, index);
  *((T *)(((uint8_t *)x.data_ptr()) + offset)) = value;
  return value;
}

////////////////////////////////////////
// 64-bit Indexing for DiffTensorView //
////////////////////////////////////////

void interlocked_add_64_bit<T, int N>(
    TensorView<T> x,
    vector<uint64_t, N> idx,
    T val,
    out T old_val,
) {
  uint64_t offset = compute_offset<T, N>(x, idx);
  let address = *((T *)(((uint8_t *)x.data_ptr()) + offset));
  interlocked_add_64_bit_inner<T>(x, offset, val, old_val);
}

[require(cuda)]
void interlocked_add_64_bit_inner<T>(
    TensorView<T> x,
    uint64_t offset,
    T val,
    out T old_val,
) {
  // Perform the atomic add in CUDA via inline PTX
  // $0 = x, $1 = offset, $2 = val, $3 = &old_val
  // clang-format off
__target_switch {
case cuda:
__intrinsic_asm "*($3) = atomicAdd(($T2*)(((uint8_t*)($0.data_ptr<$T2>())) + $1), $2)";
}
  // clang-format on
}

uint64_t compute_offset<int N>(DiffTensorView x, vector<uint64_t, N> index) {
  uint64_t offset = 0;
  uint64_t stride = sizeof(float);
  for (int i = N - 1; i >= 1; i--) {
    offset += index[i] * stride;
    stride *= (uint64_t)x.size(i);
  }
  offset += index[0] * stride;
  return offset;
}

[NoDiffThis]
float load_raw<int N>(DiffTensorView x, vector<uint64_t, N> index) {
  uint64_t offset = compute_offset(x, index);
  return *((float *)(((uint8_t *)x.primal.data_ptr()) + offset));
}

[BackwardDerivative(load_bwd)]
float load<int N>(DiffTensorView x, vector<uint64_t, N> index) {
  return load_raw(x, index);
}

void load_bwd<int N>(
    DiffTensorView x,
    vector<uint64_t, N> index,
    float.Differential d_out,
) {
  float old_val;
  interlocked_add_64_bit(x.diff.diff, index, d_out, old_val);
}

[BackwardDerivative(store_once_bwd)]
__generic<let N : int> void
store_once(DiffTensorView x, vector<uint64_t, N> index, float value) {
  store(x.primal, index, value);
}

void store_once_bwd<int N>(
    DiffTensorView x,
    vector<uint64_t, N> index,
    inout DifferentialPair<float> dp_val,
) {
  dp_val = diffPair(dp_val.p, load(x.diff.diff, index));
}
