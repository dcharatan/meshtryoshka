#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <vector>

namespace extension_cpp {
  void compute_exclusive_cumsum(const at::Tensor x) {
    TORCH_CHECK(x.dtype() == torch::kInt32);
    TORCH_CHECK(x.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(x.is_contiguous());

    int *data = x.data_ptr<int>();

    // Determine the required amount of temporary storage.
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
      nullptr,
      temp_storage_bytes,
      data,
      data,
      x.numel()
    );

    // Allocate the temporary storage.
    auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      x.options().dtype(at::kByte)
    );

    // Run the sum in place.
    cub::DeviceScan::ExclusiveSum(
      temp_storage.data_ptr(),
      temp_storage_bytes,
      data,
      data,
      x.numel()
    );
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_exclusive_cumsum", &compute_exclusive_cumsum);
  }
}
