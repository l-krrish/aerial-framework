#include "my_scale_module_kernel.cuh"

namespace framework::pipelines::samples {

__global__ void scale_module_kernel(
        const StaticKernelParams *static_params,
        const DynamicKernelParams *dynamic_params) {
    const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < static_params->size) {
        static_params->output[idx] = dynamic_params->input[idx] * static_params->scale;
    }
}

} // namespace framework::pipelines::samples