
#ifndef TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H
#define TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H

#include <vector>

#include <NvInferRuntime.h>

namespace nvinfer1
{
namespace plugin
{

class BaseCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace;
};

// Plugin factory class.
class IdentityConvCreator : public BaseCreator
{
public:
    IdentityConvCreator();

    ~IdentityConvCreator() override = default;

    char const* getPluginName() const noexcept;

    char const* getPluginVersion() const noexcept;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV3* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc,
        TensorRTPhase phase) noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;

protected:
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H
