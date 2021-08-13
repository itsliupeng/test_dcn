//
// Created by liupeng on 2021/6/22.
//

#pragma once

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "bertCommon.h"
#include "serialize.hpp"

namespace {
constexpr const char* DEFORM_CONV2D_PLUGIN_VERSION{"1"};
constexpr const char* DEFORM_CONV2D_PLUGIN_NAME{"DeformConv2D"};

nvinfer1::Weights plugin_field_to_weight(nvinfer1::PluginField field) {
  nvinfer1::Weights weight;
  weight.values = field.data;
  weight.count = field.length;
  weight.type = bert::fieldTypeToDataType(field.type);
  return weight;
}

} // namespace

namespace nvinfer1 {
namespace plugin {

class DeformConv2D : public nvinfer1::IPluginV2DynamicExt {
 private:
  // serialize
  int32_t max_batch_size = 1;
  nvinfer1::DataType dtype_;
  int32_t out_c_;
  int32_t in_c_;
  int32_t kernel_h_;
  int32_t kernel_w_;
  int32_t stride_h_;
  int32_t stride_w_;
  int32_t pad_h_;
  int32_t pad_w_;
  int32_t dilation_h_;
  int32_t dilation_w_;
  int32_t weight_groups_;
  int32_t offset_groups_;
  bool use_mask_;

  bert::WeightsWithOwnership mWeight;
  bert::WeightsWithOwnership mBias;

  //
  std::string mLayerName;
  std::string mNamespace = "";

  // device
  size_t mParamWordsize;
  bert::cuda_unique_ptr<void> mWeightDev;
  bert::cuda_unique_ptr<void> mBiasDev;
  float* mColumnDev;

  cublasHandle_t mCublas;

 protected:
  // To prevent compiler warnings
  using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::configurePlugin;
  using nvinfer1::IPluginV2DynamicExt::enqueue;
  using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
  using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::supportsFormat;

 public:
  DeformConv2D(
      nvinfer1::DataType dtype,
      nvinfer1::Weights weight,
      nvinfer1::Weights bias,
      int32_t out_c,
      int32_t in_c,
      int32_t kernel_h,
      int32_t kernel_w,
      int32_t stride_h,
      int32_t stride_w,
      int32_t pad_h,
      int32_t pad_w,
      int32_t dilation_h,
      int32_t dilation_w,
      int32_t groups,
      int32_t offset_groups,
      bool use_mask)
      : dtype_(dtype),
        out_c_(out_c),
        in_c_(in_c),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_h_(pad_h),
        pad_w_(pad_w),
        dilation_h_(dilation_h),
        dilation_w_(dilation_w),
        weight_groups_(groups),
        offset_groups_(offset_groups),
        use_mask_(use_mask),
        mColumnDev(nullptr) {
    mParamWordsize = bert::getElementSize(dtype);
    mWeight.convertAndCopy(weight, dtype);
    mBias.convertAndCopy(bias, dtype);

    bert::copyToDevice(mWeight, bert::getWeightsSize(mWeight, dtype_), mWeightDev);
    bert::copyToDevice(mBias, bert::getWeightsSize(mBias, dtype_), mBiasDev);
  }

  DeformConv2D(const std::string name, const void* data, size_t length) : mLayerName(name), mColumnDev(nullptr) {
    deserialize_value(&data, &length, &dtype_);
    deserialize_value(&data, &length, &out_c_);
    deserialize_value(&data, &length, &in_c_);
    deserialize_value(&data, &length, &kernel_h_);
    deserialize_value(&data, &length, &kernel_w_);
    deserialize_value(&data, &length, &stride_h_);
    deserialize_value(&data, &length, &stride_w_);
    deserialize_value(&data, &length, &pad_h_);
    deserialize_value(&data, &length, &pad_w_);
    deserialize_value(&data, &length, &dilation_h_);
    deserialize_value(&data, &length, &dilation_w_);
    deserialize_value(&data, &length, &weight_groups_);
    deserialize_value(&data, &length, &offset_groups_);
    int32_t mask_value;
    deserialize_value(&data, &length, &mask_value);
    use_mask_ = static_cast<bool>(mask_value);
    const char* d = static_cast<const char*>(data);
    mWeight.convertAndCopy(d, out_c_ * in_c_ * kernel_h_ * kernel_w_, dtype_);
    mBias.convertAndCopy(d, out_c_, dtype_);

    mParamWordsize = bert::getElementSize(dtype_);

    bert::copyToDevice(mWeight, bert::getWeightsSize(mWeight, dtype_), mWeightDev);
    bert::copyToDevice(mBias, bert::getWeightsSize(mBias, dtype_), mBiasDev);
  }

  DeformConv2D() = delete;

  int getNbOutputs() const override {
    return 1;
  }

  const char* getPluginType() const override {
    return DEFORM_CONV2D_PLUGIN_NAME;
  }

  const char* getPluginVersion() const override {
    return DEFORM_CONV2D_PLUGIN_VERSION;
  }

  const char* getPluginNamespace() const override {
    return mNamespace.c_str();
  }

  void setPluginNamespace(const char* pluginNamespace) override {
    mNamespace = pluginNamespace;
  }

  nvinfer1::IPluginV2DynamicExt* clone() const override {
    auto p = new DeformConv2D(
        dtype_,
        mWeight,
        mBias,
        out_c_,
        in_c_,
        kernel_h_,
        kernel_w_,
        stride_h_,
        stride_w_,
        pad_h_,
        pad_w_,
        dilation_h_,
        dilation_w_,
        weight_groups_,
        offset_groups_,
        use_mask_);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) override {
    nvinfer1::DimsExprs output(inputs[0]);

    // [n, c, h, w]
    auto out_h =
        (inputs[0].d[2]->getConstantValue() + 2 * pad_h_ - (dilation_h_ * (kernel_h_ - 1) + 1)) / stride_h_ + 1;
    auto out_w =
        (inputs[0].d[3]->getConstantValue() + 2 * pad_w_ - (dilation_w_ * (kernel_w_ - 1) + 1)) / stride_w_ + 1;

    output.d[1] = exprBuilder.constant(out_c_);
    output.d[2] = exprBuilder.constant(out_h);
    output.d[3] = exprBuilder.constant(out_w);
    return output;
  }

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override {
    return inputTypes[0];
  }

  int initialize() override {
    return 0;
  }

  void serialize(void* buffer) const {
    // attention: must be consistent with getSerializationSize
    serialize_value(&buffer, dtype_);
    serialize_value(&buffer, out_c_);
    serialize_value(&buffer, in_c_);
    serialize_value(&buffer, kernel_h_);
    serialize_value(&buffer, kernel_w_);
    serialize_value(&buffer, stride_h_);
    serialize_value(&buffer, stride_w_);
    serialize_value(&buffer, pad_h_);
    serialize_value(&buffer, pad_w_);
    serialize_value(&buffer, dilation_h_);
    serialize_value(&buffer, dilation_w_);
    serialize_value(&buffer, weight_groups_);
    serialize_value(&buffer, offset_groups_);
    serialize_value(&buffer, static_cast<int32_t>(use_mask_));

    char* d = static_cast<char*>(buffer);
    bert::serFromDev(
        d, static_cast<char*>(mWeightDev.get()), (out_c_ * in_c_ * kernel_h_ * kernel_w_) * mParamWordsize);
    bert::serFromDev(d, static_cast<char*>(mBiasDev.get()), out_c_ * mParamWordsize);
  }

  size_t getSerializationSize() const override {
    return 14 * sizeof(int32_t) + (out_c_ * in_c_ * kernel_h_ * kernel_w_) * mParamWordsize + out_c_ * mParamWordsize;
  }

  void terminate() override {}

  void destroy() override {
    if (mColumnDev) {
      gLogVerbose << "mColumnDev Free in destroy." << std::endl;
      CUASSERT(cudaFree(mColumnDev));
    }

    mWeightDev.release();
    mBiasDev.release();
    delete this;
  }

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
      override {
    ASSERT(0 <= pos && pos <= 3);
    ASSERT(nbInputs == 3);
    ASSERT(nbOutputs == 1);

    if (pos <= 2) {
      const nvinfer1::PluginTensorDesc& in = inOut[pos];
      return (in.type == (dtype_)) && (in.format == nvinfer1::TensorFormat::kLINEAR);
    }

    const nvinfer1::PluginTensorDesc& in = inOut[0];
    // pos == 3, accessing information about output tensor
    const nvinfer1::PluginTensorDesc& out = inOut[pos];

    return (out.type == (dtype_)) && (in.type == out.type) && (in.format == out.format);
  }

  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* in,
      int nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* out,
      int nbOutputs) override {}

  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc* inputs,
      int nbInputs,
      const nvinfer1::PluginTensorDesc* outputs,
      int nbOutputs) const override {
    //    auto input_shape = inputs[0].dims;
    //    auto in_c = input_shape.d[1];
    //
    //    auto output_shape = outputs[0].dims;
    //    int out_h = output_shape.d[2];
    //    int out_w = output_shape.d[3];
    //
    //    auto column_size = in_c * kernel_h_ * kernel_w_ * max_batch_size * out_h * out_w;
    //    return column_size * mParamWordsize;
    return 0;
  }

  int enqueue(
      const nvinfer1::PluginTensorDesc* inputDesc,
      const nvinfer1::PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) override;

  void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
      TRTNOEXCEPT override {
    mCublas = cublasContext;
  }
};

class DeformConv2DCreator : public nvinfer1::IPluginCreator {
 private:
  std::string mPluginNamespace;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  nvinfer1::PluginFieldCollection mFC;

 public:
  DeformConv2DCreator() : mPluginNamespace("") {
    mPluginAttributes.emplace_back(nvinfer1::PluginField("type_id", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("out_c", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("out_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("out_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("in_c", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("kernel_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("kernel_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("stride_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("stride_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("pad_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("pad_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("dilation_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("dilation_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("groups", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("offset_groups", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("use_mask", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("weight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("bias", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }

  const char* getPluginNamespace() const override {
    return mPluginNamespace.c_str();
  }

  void setPluginNamespace(const char* libNamespace) override {
    mPluginNamespace = libNamespace;
  }

  const char* getPluginName() const override {
    return DEFORM_CONV2D_PLUGIN_NAME;
  }

  const char* getPluginVersion() const override {
    return DEFORM_CONV2D_PLUGIN_VERSION;
  }

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    nvinfer1::DataType dtype;
    nvinfer1::Weights weight;
    nvinfer1::Weights bias;

    int32_t out_c;
    int32_t in_c;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t groups;
    int32_t offset_groups;
    bool use_mask;

    for (int i = 0; i < fc->nbFields; i++) {
      std::string field_name(fc->fields[i].name);
      if (field_name.compare("type_id") == 0) {
        int type_id = *static_cast<const int32_t*>(fc->fields[i].data);
        dtype = static_cast<nvinfer1::DataType>(type_id);
      } else if (field_name.compare("out_c") == 0) {
        out_c = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("in_c") == 0) {
        in_c = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("kernel_h") == 0) {
        kernel_h = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("kernel_w") == 0) {
        kernel_w = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("stride_h") == 0) {
        stride_h = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("stride_w") == 0) {
        stride_w = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("pad_h") == 0) {
        pad_h = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("pad_w") == 0) {
        pad_w = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("dilation_h") == 0) {
        dilation_h = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("dilation_w") == 0) {
        dilation_w = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("groups") == 0) {
        groups = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("offset_groups") == 0) {
        offset_groups = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("use_mask") == 0) {
        use_mask = *static_cast<const bool*>(fc->fields[i].data);
      } else if (field_name.compare("weight") == 0) {
        weight = plugin_field_to_weight(fc->fields[i]);
      } else if (field_name.compare("bias") == 0) {
        bias = plugin_field_to_weight(fc->fields[i]);
      }
    }

    return new DeformConv2D(
        dtype,
        weight,
        bias,
        out_c,
        in_c,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        offset_groups,
        use_mask);
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
    return new DeformConv2D(name, serialData, serialLength);
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return nullptr;
  }
};

} // namespace plugin
} // namespace nvinfer1
