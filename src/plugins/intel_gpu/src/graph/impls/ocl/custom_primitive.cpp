// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "custom_gpu_primitive_inst.h"
#include "jitter.h"

#include <map>
#include <sstream>
#include <vector>
#include <memory>
#include <string>

namespace kernel_selector {
using jit_constants = kernel_selector::JitConstants;
}

namespace cldnn {
namespace ocl {

struct custom_gpu_primitive_impl : typed_primitive_impl<custom_gpu_primitive> {
    using parent = typed_primitive_impl<custom_gpu_primitive>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::custom_gpu_primitive_impl)

    std::vector<std::shared_ptr<kernel_selector::cl_kernel_data>> cl_kernels;
    std::vector<kernel::ptr> _kernels;

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<custom_gpu_primitive_impl>(*this);
    }

    custom_gpu_primitive_impl()
    : _kernels() {}

    custom_gpu_primitive_impl(const custom_gpu_primitive_impl& other)
    : cl_kernels(other.cl_kernels)
    , _kernels({}) {
        for (const auto& kernel : other._kernels) {
            _kernels.emplace_back(kernel->clone(other.can_share_kernels));
        }
    }

    custom_gpu_primitive_impl(const custom_gpu_primitive_node& arg,
                             std::vector<std::shared_ptr<kernel_selector::cl_kernel_data>> cl_kernels)
        : cl_kernels(std::move(cl_kernels))
        , _kernels() { }

    std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() override {
        std::vector<std::shared_ptr<cldnn::kernel_string>> kernel_strings;
        for (const auto& cl_kernel : cl_kernels) {
            kernel_strings.push_back(cl_kernel->code.kernelString);
        }
        return kernel_strings;
    }

    void init_kernels(const kernels_cache& kernels_cache, const kernel_impl_params& params) override {
        _kernels.clear();
        auto compiled_kernels = kernels_cache.get_kernels(params);
        _kernels.insert(_kernels.begin(), compiled_kernels.begin(), compiled_kernels.end());
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    void init_by_cached_kernels(const kernels_cache& kernels_cache, std::vector<std::string>& cached_kernel_ids) override {
        _kernels.clear();
        for (const auto& kernel_id : cached_kernel_ids) {
            _kernels.emplace_back(kernels_cache.get_kernel_from_cached_kernels(kernel_id));
        }
        this->can_share_kernels = kernels_cache.get_kernels_reuse();
    }

    std::vector<std::string> get_cached_kernel_ids(const kernels_cache& kernels_cache) override {
        return {kernels_cache.get_cached_kernel_ids(_kernels)};
    }

    void set_kernels(cldnn::kernels_cache::compiled_kernels kernels) override {
        OPENVINO_ASSERT(kernels.size() == 1, "Only the kernels of the single primitive should be allowed.");
        auto& kernel_vec = kernels.begin()->second;
        _kernels.clear();
        _kernels.resize(kernel_vec.size());
        for (auto& k : kernel_vec) {
            auto sub_kernel_idx = k.second;
            _kernels[sub_kernel_idx] = k.first;
        }
    }

    void set_arguments_impl(custom_gpu_primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        for (auto& dep : instance.dependencies()) {
            args.inputs.push_back(dep.first->output_memory_ptr());
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        OPENVINO_ASSERT(_kernels.size() == cl_kernels.size(), "Custom GPU pipeline kernel count mismatch");
        for (size_t i = 0; i < _kernels.size(); ++i) {
            stream.set_arguments(*_kernels[i], cl_kernels[i]->params, args);
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                                 custom_gpu_primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        kernel_arguments_data args;
        for (auto& dep : instance.dependencies()) {
            args.inputs.push_back(dep.first->output_memory_ptr());
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        OPENVINO_ASSERT(_kernels.size() == cl_kernels.size(), "Custom GPU pipeline kernel count mismatch");
        std::vector<event::ptr> stage_events(events);
        event::ptr last_event;
        for (size_t i = 0; i < _kernels.size(); ++i) {
            const bool needs_event = (i + 1 < _kernels.size()) || instance.needs_completion_event();
            last_event = stream.enqueue_kernel(*_kernels[i], cl_kernels[i]->params, args, stage_events, needs_event);
            if (last_event) {
                stage_events = {last_event};
            }
        }
        return last_event;
    }

    std::vector<kernel::ptr> get_kernels() const override {
        return _kernels;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << cl_kernels.size();
        for (const auto& cl_kernel : cl_kernels) {
            ob << *cl_kernel;
        }
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        size_t kernel_count = 0;
        ib >> kernel_count;
        cl_kernels.clear();
        cl_kernels.reserve(kernel_count);
        for (size_t i = 0; i < kernel_count; ++i) {
            auto cl_kernel = std::make_shared<kernel_selector::cl_kernel_data>();
            ib >> *cl_kernel;
            cl_kernels.push_back(std::move(cl_kernel));
        }
    }
};

static kernel_selector::kernel_argument_element get_arg(custom_gpu_primitive::arg_desc arg) {
    kernel_selector::kernel_argument_element ret;
    switch (arg.type) {
        case custom_gpu_primitive::arg_input:
            ret.t = kernel_selector::kernel_argument_types::INPUT;
            break;
        case custom_gpu_primitive::arg_output:
            ret.t = kernel_selector::kernel_argument_types::OUTPUT;
            break;
        default:
            throw std::runtime_error("Unknown argument type");
            break;
    }

    ret.index = arg.index;

    return ret;
}

static std::string value_macro(const std::string& name, const std::string& value) {
    std::ostringstream oss;
    oss << "#define " << name << " " << value << std::endl;
    return oss.str();
}

static void add_layout_to_jit(kernel_selector::jit_constants& mem_consts, const std::string& name, const layout& l) {
    // Size (in elements)
    // #define INPUT0_DIMS (uint[]) { b, f, y, x, }
    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_DIMS", l.get_tensor().sizes(format::bfyx)));

    // Data type
    // #define INPUT0_TYPE float
    static const std::map<data_types, std::string> dataTypeToIndex{
        {data_types::i8, "char"},
        {data_types::u8, "uchar"},
        {data_types::i32, "int"},
        {data_types::i64, "long"},
        {data_types::f16, "half"},
        {data_types::f32, "float"},
    };

    OPENVINO_ASSERT(dataTypeToIndex.find(l.data_type) != dataTypeToIndex.end(), "[GPU] Add layout to jit error: unhandled data type in layout");

    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_TYPE", dataTypeToIndex.at(l.data_type)));

    // Format
    // #define INPUT0_FORMAT_BFYX
    mem_consts.AddConstant(
        kernel_selector::MakeJitConstant(name + "_FORMAT_" + kernel_selector::toString(to_data_layout(l.format)), ""));

    // Padding (in elements)
    // #define INPUT0_LOWER_PADDING (uint[]) { 0, 0, 0, 0 }
    // #define INPUT0_UPPER_PADDING (uint[]) { 0, 0, 0, 0 }
    mem_consts.AddConstant(
        kernel_selector::MakeJitConstant(name + "_LOWER_PADDING", layout::format_sizes(l.data_padding._lower_size, format::bfyx)));
    mem_consts.AddConstant(
        kernel_selector::MakeJitConstant(name + "_UPPER_PADDING", layout::format_sizes(l.data_padding._upper_size, format::bfyx)));

    // Pitches (in elements)
    // #define INPUT0_PITCHES (uint[]) { b, f, h, w, }
    // auto padded_sizes = l.get_buffer_size().sizes(format::bfyx);
    auto padded_sizes = l.get_padded_dims();

    std::vector<tensor::value_type> pitches(4);
    switch (l.format) {
        case format::bfyx:
            pitches[3] = 1;
            pitches[2] = padded_sizes[3];
            pitches[1] = padded_sizes[2] * pitches[2];
            pitches[0] = padded_sizes[1] * pitches[1];
            break;
        case format::byxf:
            pitches[1] = 1;
            pitches[3] = padded_sizes[1];
            pitches[2] = padded_sizes[3] * pitches[3];
            pitches[0] = padded_sizes[2] * pitches[2];
            break;
        case format::yxfb:
            pitches[0] = 1;
            pitches[1] = padded_sizes[0];
            pitches[3] = padded_sizes[1] * pitches[1];
            pitches[2] = padded_sizes[3] * pitches[3];
            break;
        case format::fyxb:
            pitches[0] = 1;
            pitches[3] = padded_sizes[0];
            pitches[2] = padded_sizes[3] * pitches[3];
            pitches[1] = padded_sizes[2] * pitches[2];
            break;
        default:
            throw std::runtime_error("Unhandled format in pitch calculation");
    }

    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_PITCHES", pitches));

    // Offset (in elements)
    // #define INPUT0_OFFSET 0
    auto offset =
        (pitches[0] * l.data_padding._lower_size[0]) + (pitches[1] * l.data_padding._lower_size[1]) +
        (pitches[2] * l.data_padding._lower_size[3]) + (pitches[3] * l.data_padding._lower_size[2]);
    mem_consts.AddConstant(kernel_selector::MakeJitConstant(name + "_OFFSET", std::to_string(offset)));
}

static std::string get_jit_constant(const custom_gpu_primitive_node& outer,
                                    const kernel_impl_params& impl_param,
                                    const std::vector<size_t>& gws,
                                    const std::vector<size_t>& lws) {
    kernel_selector::jit_constants mem_consts{
        kernel_selector::MakeJitConstant("NUM_INPUTS", std::to_string(outer.get_dependencies().size()))};

    mem_consts.AddConstants({
        kernel_selector::MakeJitConstant("GLOBAL_WORKSIZE", gws),
        kernel_selector::MakeJitConstant("LOCAL_WORKSIZE", lws),
    });

    for (size_t i = 0; i < impl_param.input_layouts.size(); i++) {
        add_layout_to_jit(mem_consts, "INPUT" + std::to_string(i), impl_param.get_input_layout(i));
    }

    for (size_t i = 0; i < impl_param.output_layouts.size(); i++) {
        add_layout_to_jit(mem_consts, "OUTPUT" + std::to_string(i), impl_param.get_output_layout(i));
    }

    std::ostringstream oss;
    oss << "// Custom Layer Built-ins\n\n";
    for (auto& definition : mem_consts.GetDefinitions()) {
        oss << value_macro(definition.first, definition.second);
    }

    return oss.str();
}

static std::shared_ptr<kernel_selector::cl_kernel_data> make_custom_kernel(
    const custom_gpu_primitive_node& arg,
    const kernel_impl_params& impl_param,
    const custom_gpu_primitive& primitive,
    const std::string& entry_point,
    const std::string& build_options,
    const std::vector<size_t>& gws,
    const std::vector<size_t>& lws,
    const std::vector<custom_gpu_primitive::arg_desc>& arguments,
    bool include_layout_jit = true) {
    auto cl_kernel = std::make_shared<kernel_selector::cl_kernel_data>();
    cl_kernel->code.kernelString = std::make_shared<kernel_selector::kernel_string>();
    cl_kernel->code.kernelString->entry_point = entry_point;
    cl_kernel->code.kernelString->options = build_options;
    if (include_layout_jit) {
        cl_kernel->code.kernelString->jit = get_jit_constant(arg, impl_param, gws, lws);
    }
    for (const auto& source : primitive.kernels_code) {
        cl_kernel->code.kernelString->str += source + "\n";
    }

    cl_kernel->params.workGroups.global = gws;
    cl_kernel->params.workGroups.local = lws;
    for (const auto& argument : arguments) {
        cl_kernel->params.arguments.push_back(get_arg(argument));
    }
    return cl_kernel;
}

static std::vector<std::shared_ptr<kernel_selector::cl_kernel_data>> create_intbit_projection_pipeline(
    const custom_gpu_primitive_node& arg,
    const kernel_impl_params& impl_param,
    const custom_gpu_primitive& primitive) {
    const auto input_shape = impl_param.get_input_layout(0).get_partial_shape();
    OPENVINO_ASSERT(input_shape.rank().is_static() && input_shape.size() >= 2,
                    "INTBIT projection input rank must be static and at least two");
    OPENVINO_ASSERT(input_shape[input_shape.size() - 1].is_static() &&
                    input_shape[input_shape.size() - 2].is_static(),
                    "INTBIT projection pipeline requires a concrete runtime shape");

    const size_t rows = input_shape[input_shape.size() - 2].get_length();
    const size_t kdim = input_shape[input_shape.size() - 1].get_length();
    OPENVINO_ASSERT(kdim % 128 == 0, "INTBIT K dimension must be divisible by 128, got ", kdim);
    const size_t ng = kdim / 128;
    const size_t fanout = impl_param.output_layouts.size() - 1;
    OPENVINO_ASSERT(fanout > 0, "INTBIT projection pipeline requires at least one projection output");
    OPENVINO_ASSERT(impl_param.input_layouts.size() == 2 + fanout * 3,
                    "INTBIT projection input contract is pack-x, seqb, then decode-x/weight/scale triples; got ",
                    impl_param.input_layouts.size(), " inputs for ", fanout, " projections");

    const auto packed_shape = impl_param.get_output_layout(0).get_partial_shape();
    OPENVINO_ASSERT(packed_shape.rank().is_static() && packed_shape.size() >= 2 &&
                    packed_shape[packed_shape.size() - 1].is_static() &&
                    static_cast<size_t>(packed_shape[packed_shape.size() - 1].get_length()) == kdim + kdim / 64,
                    "INTBIT packed output must have K+K/64 elements");

    std::vector<std::shared_ptr<kernel_selector::cl_kernel_data>> kernels;
    kernels.reserve(fanout + 1);
    const auto first_projection_shape = impl_param.get_output_layout(1).get_partial_shape();
    OPENVINO_ASSERT(first_projection_shape[first_projection_shape.size() - 1].is_static(),
                    "INTBIT first projection must have a static N dimension");
    const size_t first_ndim = first_projection_shape[first_projection_shape.size() - 1].get_length();
    const std::string quant_options = primitive.build_options +
        " -cl-mad-enable -DKDIM=" + std::to_string(kdim) + " -DNG=" + std::to_string(ng) +
        " -DNDIM=" + std::to_string(first_ndim) + " -DKSPLIT=4 -DINTBIT_DISABLE_GEMM";
    kernels.push_back(make_custom_kernel(
        arg,
        impl_param,
        primitive,
        "quant_pack_a8_v1",
        quant_options,
        {32, (rows - 1) * ng + 1, 1},
        {32, 1, 1},
        {{custom_gpu_primitive::arg_input, 0}, {custom_gpu_primitive::arg_output, 0}},
        false));

    for (size_t projection = 0; projection < fanout; ++projection) {
        const auto output_shape = impl_param.get_output_layout(projection + 1).get_partial_shape();
        OPENVINO_ASSERT(output_shape.rank().is_static() && output_shape.size() >= 1 &&
                        output_shape[output_shape.size() - 1].is_static(),
                        "INTBIT projection output must have a static N dimension");
        const size_t ndim = output_shape[output_shape.size() - 1].get_length();
        const std::string gemm_options = primitive.build_options +
            " -cl-mad-enable -cl-intel-256-GRF-per-thread -DKDIM=" + std::to_string(kdim) +
            " -DNG=" + std::to_string(ng) + " -DNDIM=" + std::to_string(ndim) + " -DKSPLIT=4" +
            " -DPROJECTION_INDEX=" + std::to_string(projection) + " -DINTBIT_DISABLE_QUANT";
        const auto decode_input = static_cast<custom_gpu_primitive::arg_index>(2 + projection * 3);
        const auto weight_input = static_cast<custom_gpu_primitive::arg_index>(3 + projection * 3);
        const auto scale_input = static_cast<custom_gpu_primitive::arg_index>(4 + projection * 3);
        const auto projection_output = static_cast<custom_gpu_primitive::arg_index>(projection + 1);
        kernels.push_back(make_custom_kernel(
            arg,
            impl_param,
            primitive,
            "binary_gemv_native_s2_v1",
            gemm_options,
            {ndim * 4, (rows + 127) / 128, 1},
            {64, 1, 1},
            {{custom_gpu_primitive::arg_input, decode_input},
             {custom_gpu_primitive::arg_output, 0},
             {custom_gpu_primitive::arg_input, weight_input},
             {custom_gpu_primitive::arg_input, scale_input},
             {custom_gpu_primitive::arg_input, 1},
             {custom_gpu_primitive::arg_output, projection_output}},
            false));
    }
    return kernels;
}

static std::unique_ptr<primitive_impl> create(const custom_gpu_primitive_node& arg, const kernel_impl_params& impl_param) {
    const auto primitive = arg.get_primitive().get();

    const auto& orig_output_layout = impl_param.get_output_layout();
    OPENVINO_ASSERT(orig_output_layout.is_static(), "out layouts should be static for create primitive_impl!");

    std::vector<size_t> gws, lws;
    custom_gpu_primitive::update_work_group_size(orig_output_layout.get_partial_shape(),
                                                 primitive->calcWgDimInputIdx,
                                                 orig_output_layout.get_partial_shape(),
                                                 primitive->globalSizeRules,
                                                 primitive->localSizeRules,
                                                 gws,
                                                 lws);

    if (gws.empty()) {
        gws = primitive->gws;
    }
    if (lws.empty()) {
        lws = primitive->lws;
    }

    std::vector<std::shared_ptr<kernel_selector::cl_kernel_data>> cl_kernels;
    if (primitive->kernel_entry_point == "intbit_projection_group_v1") {
        cl_kernels = create_intbit_projection_pipeline(arg, impl_param, *primitive);
    } else {
        cl_kernels.push_back(make_custom_kernel(arg,
                                                impl_param,
                                                *primitive,
                                                primitive->kernel_entry_point,
                                                primitive->build_options,
                                                gws,
                                                lws,
                                                primitive->kernel_arguments));
    }

    return std::make_unique<custom_gpu_primitive_impl>(arg, std::move(cl_kernels));
}

namespace detail {

attach_custom_gpu_primitive_impl::attach_custom_gpu_primitive_impl() {
    implementation_map<custom_gpu_primitive>::add(cldnn::impl_types::ocl, create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::custom_gpu_primitive_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::custom_gpu_primitive)
