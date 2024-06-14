#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>
#include <memory>

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include "IdentityConvPlugin.h"
#include "PluginUtils.h"

namespace nvinfer1
{
namespace plugin
{

IdentityConv::IdentityConv(IdentityConvParameters params) : mParams{params} {
    initFieldsToSerialize();
}

void IdentityConv::initFieldsToSerialize()
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(nvinfer1::PluginField(
        "kernel_shape", mParams.kernelShape.data(), PluginFieldType::kINT32, 2));
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField("strides", mParams.strides.data(), PluginFieldType::kINT32, 2));
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField("pads", mParams.pads.data(), PluginFieldType::kINT32, 4));
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField("group", &mParams.group, PluginFieldType::kINT32, 1));

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
}

// IPluginV3OneBuild methods

int32_t IdentityConv::getNbOutputs() const noexcept { return 1; }

int32_t IdentityConv::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // One output.
    PLUGIN_ASSERT(nbOutputs == getNbOutputs());
    // The output type is the same as the input type.
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t IdentityConv::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(inputs != nullptr);

    // The input tensor must be 4-D (NCHW)
    if (inputs[0].nbDims != 4)
    {
        return -1;
    }

    PLUGIN_ASSERT(nbOutputs == getNbOutputs());

    // Identity operation.
    // Just copy the dimensions from the input tensor.
    std::stringstream ss;
    ss.str("");
    ss << "Plugin info: input[0].dims = (";
    outputs[0].nbDims = inputs[0].nbDims;
    for (int i = 0; i < outputs[0].nbDims; i++)
    {
        ss << inputs[0].d[i]->getConstantValue() << ", ";
        outputs[0].d[i] = inputs[0].d[i];
    }
    ss << ")";
    logInfo(ss.str().c_str());

    return 0;
}

int32_t IdentityConv::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                                      int32_t nbInput,
                                      nvinfer1::DynamicPluginTensorDesc const* out,
                                      int32_t nbOutput) noexcept
{
    // Communicates the number of inputs and outputs, dimensions, and datatypes
    // of all inputs and outputs, broadcast information for all inputs and
    // outputs, the chosen plugin format, and maximum batch size. At this point,
    // the plugin sets up its internal state and selects the most appropriate
    // algorithm and data structures for the given configuration. Note: Resource
    // allocation is not allowed in this API because it causes a resource leak.

    // This member function will only be called during engine build time.

    // Validate input arguments.
    PLUGIN_ASSERT(nbInput == 2);
    PLUGIN_ASSERT(nbOutput == 1);
    PLUGIN_ASSERT(in[0].desc.dims.nbDims == 4);
    PLUGIN_ASSERT(out[0].desc.dims.nbDims == 4);
    for (int i = 0; i < 4; i++) {
        PLUGIN_ASSERT(in[0].desc.dims.d[i] == out[0].desc.dims.d[i]);
    }
    PLUGIN_ASSERT(in[0].desc.type == out[0].desc.type);

    mParams.dtype = in[0].desc.type;
    mParams.batchSize = in[0].desc.dims.d[0];
    mParams.channelSize = in[0].desc.dims.d[1];
    mParams.height = in[0].desc.dims.d[2];
    mParams.width = in[0].desc.dims.d[3];

    if (mParams.dtype == nvinfer1::DataType::kINT8)
    {
        mParams.dtypeBytes = 1;
    }
    else if (mParams.dtype == nvinfer1::DataType::kHALF)
    {
        mParams.dtypeBytes = 2;
    }
    else if (mParams.dtype == nvinfer1::DataType::kFLOAT)
    {
        mParams.dtypeBytes = 4;
    }
    else
    {
        PLUGIN_ASSERT(false);
    }

    return 0;
}

bool IdentityConv::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    // For this method inputs are numbered 0..(nbInputs-1) and outputs are
    // numbered nbInputs..(nbInputs+nbOutputs-1). Using this numbering, pos is
    // an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
    PLUGIN_ASSERT(nbInputs == 2 && nbOutputs == 1 &&
                  pos < nbInputs + nbOutputs);
    bool isValidCombination = false;

    // Suppose we support only a limited number of format configurations.
    isValidCombination |=
        (inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].desc.type == nvinfer1::DataType::kFLOAT);
    isValidCombination |=
        (inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].desc.type == nvinfer1::DataType::kHALF);
    // Make sure the input tensor and output tensor types and formats are same.
    isValidCombination &=
        (pos < nbInputs || (inOut[pos].desc.format == inOut[0].desc.format &&
                            inOut[pos].desc.type == inOut[0].desc.type));

    return isValidCombination;
}

// IPluginV3OneCore methods

char const* IdentityConv::getPluginName() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_NAME;
}

char const* IdentityConv::getPluginVersion() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_VERSION;
}

void IdentityConv::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* IdentityConv::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// IPluginV3 methods

nvinfer1::IPluginCapability* IdentityConv::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV3* IdentityConv::clone() noexcept
{
    auto clone = std::make_unique<IdentityConv>(*this);
    clone->initFieldsToSerialize();
    return clone.release();
}

// IPluginV3OneRuntime methods

int32_t IdentityConv::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int32_t const B = mParams.batchSize;  // inputDesc[0].dims.d[0];
    int32_t const C = mParams.channelSize;  // inputDesc[0].dims.d[1];
    int32_t const H = mParams.height;  // inputDesc[0].dims.d[2];
    int32_t const W = mParams.width;  // inputDesc[0].dims.d[3];
    int32_t const dtypeBytes = mParams.dtypeBytes;  // sizeof(float);

    size_t const inputSize{static_cast<size_t>(B * C * H * W)};
    size_t const inputSizeBytes{inputSize * dtypeBytes};
    cudaError_t const status{cudaMemcpyAsync(outputs[0], inputs[0],
                                             inputSizeBytes,
                                             cudaMemcpyDeviceToDevice, stream)};

    return status;
}

int32_t IdentityConv::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

nvinfer1::IPluginV3* IdentityConv::attachToContext(nvinfer1::IPluginResourceContext* context) noexcept
{
    return clone();
}

nvinfer1::PluginFieldCollection const* IdentityConv::getFieldsToSerialize() noexcept
{
    return &mFCToSerialize;
}

size_t IdentityConv::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // No scratch space is required for this plugin.
    return 0;
}

} // namespace plugin
} // namespace nvinfer1