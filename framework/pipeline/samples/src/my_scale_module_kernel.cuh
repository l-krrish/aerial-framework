#ifndef MY_SCALE_MODULE_KERNEL_CUH
#define MY_SCALE_MODULE_KERNEL_CUH

#include <cstddef>

namespace framework::pipelines::samples {

struct StaticKernelParams final {
    float *output{nullptr};
    std::size_t size{0};
    float scale{1.0F};
};

struct DynamicKernelParams final {
    const float *input{nullptr};
};

__global__ void scale_module_kernel(
        const StaticKernelParams *static_params,
        const DynamicKernelParams *dynamic_params);

} // namespace framework::pipelines::samples

#endif // MY_SCALE_MODULE_KERNEL_CUH