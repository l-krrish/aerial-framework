#include "my_scale_module.hpp"
#include <format>
#include <stdexcept>
#include <utility>

#include "log/rt_log_macros.hpp"
#include "pipeline/igraph.hpp"
#include "tensor/data_types.hpp"
#include "utils/error_macros.hpp"


namespace framework::pipelines::samples {

// Namespace alias for compatibility with framework reorganization
namespace pipeline = ::framework::pipeline;
namespace tensor = ::framework::tensor;

// ============================================================================
// Construction
// ============================================================================

ScaleModule::ScaleModule(std::string instance_id, const StaticParams &params)
        : instance_id_(std::move(instance_id)), tensor_size_(params.tensor_size),
          tensor_bytes_(params.tensor_size * sizeof(float)),
          scale_(params.scale),
          execution_mode_(params.execution_mode) {

    RT_LOG_INFO(
            "ScaleModule: Constructing instance '{}', tensor_size={}, "
            "execution_mode={}",
            instance_id_,
            tensor_size_,
            execution_mode_ == pipeline::ExecutionMode::Graph ? "Graph" : "Stream");

    if (tensor_size_ == 0) {
        const std::string error_msg =
                std::format("ScaleModule '{}': tensor_size cannot be zero", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    // Calculate grid size for kernel launches
    grid_size_ = (tensor_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

    RT_LOG_DEBUG("ScaleModule '{}': Constructor complete", instance_id_);
}

// ============================================================================
// Interface Access
// ============================================================================

pipeline::IStreamExecutor *ScaleModule::as_stream_executor() { return this; }

pipeline::IGraphNodeProvider *ScaleModule::as_graph_node_provider() { return this; }

// ============================================================================
// Port Introspection (called during pipeline construction)
// ============================================================================

std::vector<std::string> ScaleModule::get_input_port_names() const { return {"input"}; }

std::vector<std::string> ScaleModule::get_output_port_names() const { return {"output"}; }

std::vector<tensor::TensorInfo>
ScaleModule::get_input_tensor_info(std::string_view port_name) const {
    if (port_name != "input") {
        const std::string error_msg =
                std::format("ScaleModule '{}': Unknown input port '{}'", instance_id_, port_name);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    return {tensor::TensorInfo(tensor::TensorInfo::DataType::TensorR32F, {tensor_size_})};
}

std::vector<tensor::TensorInfo>
ScaleModule::get_output_tensor_info(std::string_view port_name) const {
    if (port_name != "output") {
        const std::string error_msg = std::format(
                "ScaleModule '{}': Unknown output port '{}'", instance_id_, port_name);
        RT_LOG_ERROR("{}", error_msg);
        throw std::invalid_argument(error_msg);
    }

    return {tensor::TensorInfo(tensor::TensorInfo::DataType::TensorR32F, {tensor_size_})};
}

// ============================================================================
// Memory Configuration (called before setup())
// ============================================================================

pipeline::InputPortMemoryCharacteristics
ScaleModule::get_input_memory_characteristics(std::string_view port_name) const {
    if (port_name == "input") {
        return pipeline::InputPortMemoryCharacteristics{
                .requires_fixed_address_for_zero_copy = false};
    }

    const std::string error_msg =
            std::format("ScaleModule '{}': Unknown input port '{}'", instance_id_, port_name);
    RT_LOG_ERROR("{}", error_msg);
    throw std::invalid_argument(error_msg);
}

pipeline::OutputPortMemoryCharacteristics
ScaleModule::get_output_memory_characteristics(std::string_view port_name) const {
    if (port_name == "output") {
        return pipeline::OutputPortMemoryCharacteristics{
                .provides_fixed_address_for_zero_copy = true};
    }

    const std::string error_msg =
            std::format("ScaleModule '{}': Unknown output port '{}'", instance_id_, port_name);
    RT_LOG_ERROR("{}", error_msg);
    throw std::invalid_argument(error_msg);
}

pipeline::ModuleMemoryRequirements ScaleModule::get_requirements() const {
    pipeline::ModuleMemoryRequirements reqs{};

    reqs.static_kernel_descriptor_bytes = sizeof(StaticKernelParams);
    reqs.dynamic_kernel_descriptor_bytes = sizeof(DynamicKernelParams);

    reqs.device_tensor_bytes = tensor_bytes_;
    reqs.alignment = MEMORY_ALIGNMENT;

    RT_LOG_DEBUG(
            "ScaleModule '{}': Memory requirements - static_desc={}, "
            "dynamic_desc={}, device={} bytes",
            instance_id_,
            reqs.static_kernel_descriptor_bytes,
            reqs.dynamic_kernel_descriptor_bytes,
            reqs.device_tensor_bytes);

    return reqs;
}

// ============================================================================
// Setup Phase (one-time initialization)
// ============================================================================

void ScaleModule::setup_memory(const pipeline::ModuleMemorySlice &memory_slice) {
    RT_LOG_INFO(
            "ScaleModule '{}': setup_memory() called - static_gpu={}, "
            "dynamic_gpu={}, tensor={}",
            instance_id_,
            static_cast<void *>(memory_slice.static_kernel_descriptor_gpu_ptr),
            static_cast<void *>(memory_slice.dynamic_kernel_descriptor_gpu_ptr),
            static_cast<void *>(memory_slice.device_tensor_ptr));

    mem_slice_ = memory_slice;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    d_output_ = reinterpret_cast<float *>(mem_slice_.device_tensor_ptr);

    RT_LOG_DEBUG(
            "ScaleModule '{}': Allocated output tensor at {}",
            instance_id_,
            static_cast<void *>(d_output_));

    kernel_desc_mgr_ = std::make_unique<pipeline::KernelDescriptorAccessor>(memory_slice);

    static_params_cpu_ptr_ =
            &kernel_desc_mgr_->create_static_param<StaticKernelParams>(0);

    static_params_cpu_ptr_->output = d_output_;
    static_params_cpu_ptr_->size = tensor_size_;
    static_params_cpu_ptr_->scale = scale_;

    dynamic_params_cpu_ptr_ =
            &kernel_desc_mgr_->create_dynamic_param<DynamicKernelParams>(0);

    dynamic_params_cpu_ptr_->input = nullptr;

    static_params_gpu_ptr_ =
            kernel_desc_mgr_->get_static_device_ptr<StaticKernelParams>(0);
    dynamic_params_gpu_ptr_ =
            kernel_desc_mgr_->get_dynamic_device_ptr<DynamicKernelParams>(0);

    RT_LOG_INFO(
            "ScaleModule '{}': Kernel descriptors initialized - "
            "static_cpu={}, static_gpu={}, dynamic_cpu={}, dynamic_gpu={}",
            instance_id_,
            static_cast<void *>(static_params_cpu_ptr_),
            static_cast<void *>(static_params_gpu_ptr_),
            static_cast<void *>(dynamic_params_cpu_ptr_),
            static_cast<void *>(dynamic_params_gpu_ptr_));

    pipeline::setup_kernel_function(
            kernel_config_,
            reinterpret_cast<const void *>(scale_module_kernel));

    pipeline::setup_kernel_dimensions(
            kernel_config_,
            dim3(static_cast<unsigned int>(grid_size_), 1, 1),
            dim3(BLOCK_SIZE, 1, 1));

    pipeline::setup_kernel_arguments(
            kernel_config_, static_params_gpu_ptr_, dynamic_params_gpu_ptr_);

    RT_LOG_INFO(
            "ScaleModule '{}': Kernel launch configuration complete - "
            "grid={}, block={}",
            instance_id_,
            grid_size_,
            BLOCK_SIZE);
}

void ScaleModule::set_inputs(std::span<const pipeline::PortInfo> inputs) {
    RT_LOG_DEBUG(
            "ScaleModule '{}': set_inputs() called with {} ports", instance_id_, inputs.size());

    for (const auto &port : inputs) {
        if (port.tensors.empty()) {
            const std::string error_msg = std::format(
                    "ScaleModule '{}': Port '{}' has no tensors", instance_id_, port.name);
            RT_LOG_ERROR("{}", error_msg);
            throw std::invalid_argument(error_msg);
        }

        if (port.name == "input") {
            d_input_ = port.tensors[0].device_ptr;
            RT_LOG_INFO(
                    "ScaleModule '{}': set_inputs() - Set d_input_={}", instance_id_, d_input_);
        } else {
            const std::string error_msg = std::format(
                    "ScaleModule '{}': Unknown input port '{}'", instance_id_, port.name);
            RT_LOG_ERROR("{}", error_msg);
            throw std::invalid_argument(error_msg);
        }
    }
}

void ScaleModule::warmup([[maybe_unused]] cudaStream_t stream) {
    RT_LOG_INFO(
            "ScaleModule '{}': warmup(stream={}) called - no warmup "
            "required for simple CUDA kernel",
            instance_id_,
            static_cast<void *>(stream));
}

// ============================================================================
// Per-Iteration Configuration
// ============================================================================

void ScaleModule::configure_io(
        const pipeline::DynamicParams &params, [[maybe_unused]] cudaStream_t stream) {
    RT_LOG_INFO(
            "ScaleModule '{}': configure_io() - d_input_={}, "
            "updating dynamic_params_cpu_ptr_->input",
            instance_id_,
            d_input_);

    (void)params;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    dynamic_params_cpu_ptr_->input = reinterpret_cast<const float *>(d_input_);

    RT_LOG_INFO(
            "ScaleModule '{}': Dynamic params CPU: input={} (should match "
            "d_input_={}). Pipeline will bulk-copy to device.",
            instance_id_,
            static_cast<const void *>(dynamic_params_cpu_ptr_->input),
            d_input_);
}

std::vector<pipeline::PortInfo> ScaleModule::get_outputs() const {
    const tensor::TensorInfo output_info(tensor::TensorInfo::DataType::TensorR32F, {tensor_size_});
    const pipeline::DeviceTensor output_tensor{.device_ptr = d_output_, .tensor_info = output_info};
    return {pipeline::PortInfo{.name = "output", .tensors = {output_tensor}}};
}

// ============================================================================
// Execution - Stream Mode
// ============================================================================

void ScaleModule::execute(cudaStream_t stream) {
    RT_LOG_DEBUG(
            "ScaleModule '{}': execute() on stream {}",
            instance_id_,
            static_cast<void *>(stream));

    launch_scale_kernel(stream);
}

// ============================================================================
// Execution - Graph Mode
// ============================================================================

std::span<const CUgraphNode> ScaleModule::add_node_to_graph(
        gsl_lite::not_null<pipeline::IGraph *> graph, std::span<const CUgraphNode> deps) {
    if (d_input_ == nullptr) {
        const std::string error_msg =
                std::format("ScaleModule '{}': Input not set before graph capture", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOG_DEBUG(
            "ScaleModule '{}': Adding kernel node to graph with {} dependencies",
            instance_id_,
            deps.size());

    kernel_node_ = graph->add_kernel_node(deps, kernel_config_.get_kernel_params());

    RT_LOG_DEBUG(
            "ScaleModule '{}': Kernel node added: {}",
            instance_id_,
            static_cast<void *>(kernel_node_));

    return {&kernel_node_, 1};
}

void ScaleModule::update_graph_node_params(
        CUgraphExec exec, [[maybe_unused]] const pipeline::DynamicParams &params) {
    const auto &kernel_params = kernel_config_.get_kernel_params();
    FRAMEWORK_CUDA_DRIVER_CHECK_THROW(
            cuGraphExecKernelNodeSetParams(exec, kernel_node_, &kernel_params));

    RT_LOG_DEBUG("ScaleModule '{}': Graph node params updated", instance_id_);
}

// ============================================================================
// Private Helpers
// ============================================================================

void ScaleModule::launch_scale_kernel(cudaStream_t stream) {
    if (d_input_ == nullptr) {
        const std::string error_msg =
                std::format("ScaleModule '{}': Input not set before execution", instance_id_);
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOG_DEBUG(
            "ScaleModule '{}': Launching scale kernel on stream {}",
            instance_id_,
            static_cast<void *>(stream));

    const CUresult launch_err = kernel_config_.launch(stream);

    if (launch_err != CUDA_SUCCESS) {
        const char *error_str = nullptr;
        cuGetErrorString(launch_err, &error_str);
        const std::string error_msg = std::format(
                "ScaleModule '{}': Kernel launch failed: {} ({})",
                instance_id_,
                static_cast<int>(launch_err),
                error_str != nullptr ? error_str : "unknown");
        RT_LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    RT_LOG_DEBUG("ScaleModule '{}': Scale kernel launched", instance_id_);
}

} // namespace framework::pipelines::samples