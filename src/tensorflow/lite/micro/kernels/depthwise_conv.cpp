/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/kernels/depthwise_conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "esp_nn.h"

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  const OpDataConv& data = *(static_cast<const OpDataConv*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kDepthwiseConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kDepthwiseConvBiasTensor)
          : nullptr;

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::DepthwiseConv(
          DepthwiseConvParamsFloat(params, data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    }
    case kTfLiteInt8: {
      const auto& input_shape = tflite::micro::GetTensorShape(input);
      const auto& filter_shape = tflite::micro::GetTensorShape(filter);
      const auto& output_shape = tflite::micro::GetTensorShape(output);
      const auto dw_params = DepthwiseConvParamsQuantized(params, data);

      data_dims_t input_dims = {input_shape.Dims(2), input_shape.Dims(1),
                                input_shape.Dims(3), 0};
      data_dims_t filter_dims = {filter_shape.Dims(2), filter_shape.Dims(1),
                                 filter_shape.Dims(3), 0};
      data_dims_t output_dims = {output_shape.Dims(2), output_shape.Dims(1),
                                 output_shape.Dims(3), 0};
      dw_conv_params_t esp_dw_params = {};
      esp_dw_params.in_offset = dw_params.input_offset;
      esp_dw_params.out_offset = dw_params.output_offset;
      esp_dw_params.ch_mult = dw_params.depth_multiplier;
      esp_dw_params.stride = {dw_params.stride_width, dw_params.stride_height};
      esp_dw_params.padding = {dw_params.padding_values.width,
                               dw_params.padding_values.height};
      esp_dw_params.dilation = {dw_params.dilation_width_factor,
                                dw_params.dilation_height_factor};
      esp_dw_params.activation = {dw_params.quantized_activation_min,
                                  dw_params.quantized_activation_max};
      quant_data_t quant = {data.per_channel_output_shift,
                            data.per_channel_output_multiplier};

      int scratch_size = esp_nn_get_depthwise_conv_scratch_size(
          &input_dims, &filter_dims, &output_dims, &esp_dw_params);
      if (scratch_size > 0) {
        void* scratch = __builtin_alloca(scratch_size);
        esp_nn_set_depthwise_conv_scratch_buf(scratch);
      }

      esp_nn_depthwise_conv_s8(&input_dims,
                               tflite::micro::GetTensorData<int8_t>(input),
                               &filter_dims,
                               tflite::micro::GetTensorData<int8_t>(filter),
                               tflite::micro::GetOptionalTensorData<int32_t>(bias),
                               &output_dims,
                               tflite::micro::GetTensorData<int8_t>(output),
                               &esp_dw_params, &quant);
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_DEPTHWISE_CONV_2D() {
  return tflite::micro::RegisterOp(Init, DepthwiseConvPrepare, Eval);
}

}  // namespace tflite
