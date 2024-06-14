#ifndef TENSORRT_IDENTITY_CONV_PLUGIN_H
#define TENSORRT_IDENTITY_CONV_PLUGIN_H

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <NvInfer.h>
//#include <NvInferRuntimePlugin.h>

constexpr char const* const kIDENTITY_CONV_PLUGIN_NAME{"IdentityConv"};
constexpr char const* const kIDENTITY_CONV_PLUGIN_VERSION{"1"};

namespace nvinfer1
{
namespace plugin
{

struct IdentityConvParameters
{
    std::vector<int32_t> kernelShape;
    std::vector<int32_t> strides;
    std::vector<int32_t> pads;
    int32_t group;
    nvinfer1::DataType dtype;
    int32_t batchSize;
    int32_t channelSize;
    int32_t height;
    int32_t width;
    size_t dtypeBytes;
};

class IdentityConv : public nvinfer1::IPluginV3, public nvinfer1::IPluginV3OneCore, public nvinfer1::IPluginV3OneBuild, public nvinfer1::IPluginV3OneRuntime
{
public:
//    IdentityConv() = default;
//
//    IdentityConv(IdentityConv const& p) = default;
//
//    IdentityConv& operator=(IdentityConv const& p) = default;

    IdentityConv(IdentityConvParameters params);

    ~IdentityConv() override = default;

    void initFieldsToSerialize();

    // IPluginV3OneBuild methods
    int32_t getNbOutputs() const noexcept override;

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInput,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutput) noexcept override;

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept;

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV3 methods
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;

    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneRuntime methods
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

private:
    // TensorRT plugin parameters.
    IdentityConvParameters mParams;

    nvinfer1::PluginFieldCollection mFCToSerialize;
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TENSORRT_IDENTITY_CONV_PLUGIN_H
