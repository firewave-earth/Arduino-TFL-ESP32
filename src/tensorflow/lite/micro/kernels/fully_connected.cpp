/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/fully_connected.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "esp_nn.h"

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataFullyConnected));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  TF_LITE_ENSURE_OK(context, CalculateOpDataFullyConnected(
                                 context, params->activation, input->type,
                                 input, filter, bias, output, data));

  // The converter quantizes FULLY_CONNECTED weights per channel by default.
  // The single multiplier above uses only channel 0's scale, so per-channel
  // weights need one multiplier per output channel.
  data->per_channel_output_multiplier = nullptr;
  data->per_channel_output_shift = nullptr;
  const auto* affine_quantization =
      static_cast<const TfLiteAffineQuantization*>(filter->quantization.params);
  if (filter->quantization.type == kTfLiteAffineQuantization &&
      affine_quantization != nullptr && affine_quantization->scale != nullptr &&
      affine_quantization->scale->size > 1) {
    if (input->type != kTfLiteInt8) {
      TF_LITE_KERNEL_LOG(context,
                         "Per-channel FULLY_CONNECTED weights are supported "
                         "only for int8 input, not %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
    TF_LITE_ENSURE_EQ(context, affine_quantization->quantized_dimension, 0);
    const int out_channels = filter->dims->data[0];
    data->per_channel_output_multiplier =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, out_channels * sizeof(int32_t)));
    data->per_channel_output_shift =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, out_channels * sizeof(int32_t)));
    TF_LITE_ENSURE(context, data->per_channel_output_multiplier != nullptr &&
                                data->per_channel_output_shift != nullptr);
    TF_LITE_ENSURE_STATUS(PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier, data->per_channel_output_shift,
        out_channels));
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data =
      *(static_cast<const OpDataFullyConnected*>(node->user_data));

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32: {
      const float* bias_data =
          nullptr != bias ? tflite::micro::GetTensorData<float>(bias) : nullptr;

      tflite::reference_ops::FullyConnected(
          FullyConnectedParamsFloat(params->activation),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias), bias_data,
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    }

    case kTfLiteInt8: {
      const auto& filter_shape = tflite::micro::GetTensorShape(filter);
      const auto fc_params = FullyConnectedParamsQuantized(data);

      const int32_t* bias_data =
          nullptr != bias ? tflite::micro::GetTensorData<int32_t>(bias)
                          : nullptr;
      int out_channels = filter_shape.Dims(0);
      int row_len = filter_shape.FlatSize() / out_channels;

      if (data.per_channel_output_multiplier != nullptr) {
        esp_nn_fully_connected_per_ch_s8(
            tflite::micro::GetTensorData<int8_t>(input),
            fc_params.input_offset,
            row_len,
            tflite::micro::GetTensorData<int8_t>(filter),
            fc_params.weights_offset,
            bias_data,
            tflite::micro::GetTensorData<int8_t>(output),
            out_channels,
            fc_params.output_offset,
            data.per_channel_output_shift,
            data.per_channel_output_multiplier,
            fc_params.quantized_activation_min,
            fc_params.quantized_activation_max);
        break;
      }

      esp_nn_fully_connected_s8(
          tflite::micro::GetTensorData<int8_t>(input),
          fc_params.input_offset,
          row_len,
          tflite::micro::GetTensorData<int8_t>(filter),
          fc_params.weights_offset,
          bias_data,
          tflite::micro::GetTensorData<int8_t>(output),
          out_channels,
          fc_params.output_offset,
          fc_params.output_shift,
          fc_params.output_multiplier,
          fc_params.quantized_activation_min,
          fc_params.quantized_activation_max);
      break;
    }

    case kTfLiteInt16: {
      const int64_t* bias_data =
          nullptr != bias ? tflite::micro::GetTensorData<int64_t>(bias)
                          : nullptr;

      tflite::reference_integer_ops::FullyConnected(
          FullyConnectedParamsQuantized(data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(bias), bias_data,
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
      break;
    }

    default: {
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
