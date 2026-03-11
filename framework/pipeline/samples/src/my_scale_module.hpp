#ifndef MY_SCALE_MODULE_HPP
#define MY_SCALE_MODULE_HPP

#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <cstddef>

#include <cuda.h>
#include <driver_types.h>
#include <gsl-lite/gsl-lite.hpp>

#include "pipeline/types.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/kernel_launch_config.hpp"
#include "pipeline/kernel_descriptor_accessor.hpp"
#include "tensor/tensor_info.hpp"
#include "my_scale_module_kernel.cuh"

namespace framework::pipelines::samples {

namespace pipeline = ::framework::pipeline;
namespace tensor = ::framework::tensor;

class ScaleModule final : public pipeline::IModule,
                          public pipeline::IAllocationInfoProvider,
                          public pipeline::IGraphNodeProvider,
                          public pipeline::IStreamExecutor {
public:
    struct StaticParams {
        std::size_t tensor_size{0};
        float scale{1.0F};
        pipeline::ExecutionMode execution_mode{pipeline::ExecutionMode::Graph};
    };

    ScaleModule(std::string instance_id, const StaticParams &params);
    ~ScaleModule() override = default;

    ScaleModule(const ScaleModule &) = delete;
    ScaleModule &operator=(const ScaleModule &) = delete;
    ScaleModule(ScaleModule &&) = delete;
    ScaleModule &operator=(ScaleModule &&) = delete;

    // ========================================================================
    // IModule Interface - Identification
    // ========================================================================

    [[nodiscard]] std::string_view get_type_id() const override { return "scale_module"; }

    [[nodiscard]] std::string_view get_instance_id() const override { return instance_id_; }

    [[nodiscard]] pipeline::IStreamExecutor *as_stream_executor() override;

    [[nodiscard]] pipeline::IGraphNodeProvider *as_graph_node_provider() override;

    // ========================================================================
    // IModule Interface - Port Introspection
    // ========================================================================

    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;

    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;

    [[nodiscard]] std::vector<tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;

    [[nodiscard]] std::vector<tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;

    // ========================================================================
    // IModule & IAllocationInfoProvider - Memory Configuration
    // ========================================================================

    [[nodiscard]] pipeline::InputPortMemoryCharacteristics
    get_input_memory_characteristics(std::string_view port_name) const override;

    [[nodiscard]] pipeline::OutputPortMemoryCharacteristics
    get_output_memory_characteristics(std::string_view port_name) const override;

    [[nodiscard]] pipeline::ModuleMemoryRequirements get_requirements() const override;

    // ========================================================================
    // IModule Interface - Setup Phase
    // ========================================================================

    void setup_memory(const pipeline::ModuleMemorySlice &memory_slice) override;

    void set_inputs(std::span<const pipeline::PortInfo> inputs) override;

    void warmup(cudaStream_t stream) override;

    // ========================================================================
    // IModule Interface - Per-Iteration Configuration
    // ========================================================================

    void configure_io(const pipeline::DynamicParams &params, cudaStream_t stream) override;

    [[nodiscard]] std::vector<pipeline::PortInfo> get_outputs() const override;

    // ========================================================================
    // IStreamExecutor Interface - Stream Mode Execution
    // ========================================================================

    void execute(cudaStream_t stream) override;

    // ========================================================================
    // IGraphNodeProvider Interface - Graph Mode Execution
    // ========================================================================

    [[nodiscard]] std::span<const CUgraphNode> add_node_to_graph(
            gsl_lite::not_null<pipeline::IGraph *> graph,
            std::span<const CUgraphNode> deps) override;

    void update_graph_node_params(CUgraphExec exec, const pipeline::DynamicParams &params) override;

private:
    void launch_scale_kernel(cudaStream_t stream);

    std::string instance_id_;
    std::size_t tensor_size_;
    std::size_t tensor_bytes_;
    float scale_;
    pipeline::ExecutionMode execution_mode_;
    pipeline::ModuleMemorySlice mem_slice_;

    std::unique_ptr<pipeline::KernelDescriptorAccessor> kernel_desc_mgr_;
    StaticKernelParams *static_params_cpu_ptr_{nullptr};
    DynamicKernelParams *dynamic_params_cpu_ptr_{nullptr};
    StaticKernelParams *static_params_gpu_ptr_{nullptr};
    DynamicKernelParams *dynamic_params_gpu_ptr_{nullptr};

    float *d_output_{nullptr};
    const void *d_input_{nullptr};
    CUgraphNode kernel_node_{nullptr};
    pipeline::DualKernelLaunchConfig kernel_config_;

    static constexpr unsigned int BLOCK_SIZE = 256;
    static constexpr std::size_t MEMORY_ALIGNMENT = 256;
    std::size_t grid_size_{0};
};

} // namespace framework::pipelines::samples

#endif // MY_SCALE_MODULE_HPP
