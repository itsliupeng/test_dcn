#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto deform_conv2d_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"torchvision::deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensor();
       auto weight = args[1].IValue()->toTensor();
       auto offset = args[2].ITensor();
       auto mask = args[3].ITensor();
       auto bias = args[4].IValue()->toTensor();
       ;
       auto stride_h = args[5].unwrapToInt();
       auto stride_w = args[6].unwrapToInt();
       auto pad_h = args[7].unwrapToInt();
       auto pad_w = args[8].unwrapToInt();
       auto dilation_h = args[9].unwrapToInt();
       auto dilation_w = args[10].unwrapToInt();
       auto groups = args[11].unwrapToInt();
       auto offset_groups = args[12].unwrapToInt();
       auto use_mask = args[13].unwrapToBool();

       auto weight_shape = weight.sizes();
       auto weight_count = std::accumulate(weight_shape.begin(), weight_shape.end(), 1, std::multiplies<int64_t>());
       LOG_DEBUG("weight shape: " << weight_shape);
       TORCH_CHECK(weight_shape.size() == 4, "weight shape should be [out_c, in_c, kernel_h, kernel_w]")

       auto out_c = weight_shape[0];
       auto in_c = weight_shape[1];
       auto kernel_h = weight_shape[2];
       auto kernel_w = weight_shape[3];

       nvinfer1::PluginFieldCollection fc;
       std::vector<nvinfer1::PluginField> f;

       int type_id =
           ctx->settings.enabled_precisions.find(nvinfer1::DataType::kHALF) == ctx->settings.enabled_precisions.end()
           ? 0
           : 1; // Integer encoding the DataType (0: FP32, 1: FP16)
       f.emplace_back(nvinfer1::PluginField("type_id", &type_id, nvinfer1::PluginFieldType::kINT32, 1));
       // weight
       f.emplace_back(nvinfer1::PluginField("out_c", &out_c, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(nvinfer1::PluginField("in_c", &in_c, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(nvinfer1::PluginField("kernel_h", &kernel_h, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(nvinfer1::PluginField("kernel_w", &kernel_w, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(
           nvinfer1::PluginField("weight", weight.data_ptr(), nvinfer1::PluginFieldType::kFLOAT32, weight_count));
       // bias
       f.emplace_back(
           nvinfer1::PluginField("bias", bias.data_ptr(), nvinfer1::PluginFieldType::kFLOAT32, bias.size(0)));
       f.emplace_back(nvinfer1::PluginField("stride_h", &stride_h, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(nvinfer1::PluginField("stride_w", &stride_w, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(nvinfer1::PluginField("pad_h", &pad_h, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(nvinfer1::PluginField("pad_w", &pad_w, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(nvinfer1::PluginField("dilation_h", &dilation_h, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(nvinfer1::PluginField("dilation_w", &dilation_w, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(nvinfer1::PluginField("groups", &groups, nvinfer1::PluginFieldType::kINT32, 1));
       f.emplace_back(nvinfer1::PluginField("offset_groups", &offset_groups, nvinfer1::PluginFieldType::kINT32, 1));
       auto mask_value = use_mask ? 1 : 0;
       f.emplace_back(nvinfer1::PluginField("use_mask", &mask_value, nvinfer1::PluginFieldType::kINT32, 1));
       fc.nbFields = f.size();
       fc.fields = f.data();

       auto creator = getPluginRegistry()->getPluginCreator("DeformConv2D", "1", "");
       auto plugin = creator->createPlugin("DeformConv2D", &fc);

       std::vector<nvinfer1::ITensor*> in_list = {in, offset, mask};
       auto layer = ctx->net->addPluginV2(in_list.data(), in_list.size(), *plugin);
       TRTORCH_CHECK(layer, "Unable to create DeformConv2D plugin from node " << *n);
       layer->setName(util::node_info(n).c_str());
       auto dcn_output = layer->getOutput(0);

       // [c, b, h, w] => [b, c, h, w]
       auto reshape_layer = ctx->net->addShuffle(*dcn_output);
       nvinfer1::Permutation permute;
       std::vector<int32_t> new_order = {1, 0, 2, 3};
       std::copy(new_order.begin(), new_order.end(), permute.order);
       reshape_layer->setSecondTranspose(permute);
       auto reshape_output = reshape_layer->getOutput(0);

       auto out = ctx->AssociateValueAndTensor(n->outputs()[0], reshape_output);

       LOG_DEBUG("Output tensor shape: " << out->getDimensions());

       return
           true;
     }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
