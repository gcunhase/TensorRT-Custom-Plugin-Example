#include <exception>

#include <NvInferRuntimePlugin.h>

#include "IdentityConvPlugin.h"
#include "IdentityConvPluginCreator.h"
#include "PluginUtils.h"

// Plugin creator
IdentityConvCreator::IdentityConvCreator() {}

char const* IdentityConvCreator::getPluginName() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_NAME;
}

char const* IdentityConvCreator::getPluginVersion() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const*
IdentityConvCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2IOExt* IdentityConvCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    // The attributes from the ONNX node will be parsed and passed via fc.
    // In our dummy case,
    // attrs={
    //     "kernel_shape": [1, 1],
    //     "strides": [1, 1],
    //     "pads": [0, 0, 0, 0],
    //     "group": num_groups
    // }

    try
    {
        nvinfer1::PluginField const* fields{fc->fields};
        int32_t nbFields{fc->nbFields};

        PLUGIN_VALIDATE(nbFields == 4);

        std::vector<int32_t> kernelShape{};
        std::vector<int32_t> strides{};
        std::vector<int32_t> pads{};
        int32_t group{};

        for (int32_t i{0}; i < nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "kernel_shape"))
            {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                int32_t const* const kernelShapeData{
                    static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j)
                {
                    kernelShape.push_back(kernelShapeData[j]);
                }
            }
            if (!strcmp(attrName, "strides"))
            {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                int32_t const* const stridesData{
                    static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j)
                {
                    strides.push_back(stridesData[j]);
                }
            }
            if (!strcmp(attrName, "pads"))
            {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                int32_t const* const padsData{
                    static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j)
                {
                    pads.push_back(padsData[j]);
                }
            }
            if (!strcmp(attrName, "group"))
            {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                PLUGIN_VALIDATE(fields[i].length == 1);
                group = *(static_cast<int32_t const*>(fields[i].data));
            }
        }

        IdentityConvParameters const params{.group = group};

        IdentityConv* const plugin{new IdentityConv{params}};
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV2IOExt*
IdentityConvCreator::deserializePlugin(char const* name, void const* serialData,
                                       size_t serialLength) noexcept
{
    try
    {
        IdentityConv* plugin = new IdentityConv{serialData, serialLength};
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
