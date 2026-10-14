/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "esp_nn.h"

#include <limits>

namespace tflite {

namespace {

// int8 MAX_POOL_2D on esp-nn, one batch at a time, when esp-nn computes exactly
// what MaxPoolingEvalQuantized would; returns false, doing nothing, otherwise.
// esp-nn's S3 kernel needs a channel count that is a multiple of 4, takes every
// size as uint16_t, and up to esp-nn ce8dc57 skips activation_min unless the
// channel count is a multiple of 8.
bool MaxPoolingEvalEspNn(const TfLitePoolParams* params,
                         const OpDataPooling* data,
                         const TfLiteEvalTensor* input,
                         TfLiteEvalTensor* output) {
  const RuntimeShape input_shape = micro::GetTensorShape(input);
  const RuntimeShape output_shape = micro::GetTensorShape(output);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  if (depth % 4 != 0) return false;
  if (depth % 8 != 0 &&
      data->activation_min != std::numeric_limits<int8_t>::min()) {
    return false;
  }
  const int sizes[] = {input_shape.Dims(1),    input_shape.Dims(2),
                       output_shape.Dims(1),   output_shape.Dims(2),
                       params->stride_height,  params->stride_width,
                       params->filter_height,  params->filter_width,
                       data->padding.height,   data->padding.width,
                       depth};
  for (int size : sizes) {
    if (size < 0 || size > std::numeric_limits<uint16_t>::max()) return false;
  }

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_batch_size =
      input_shape.Dims(1) * input_shape.Dims(2) * depth;
  const int output_batch_size =
      output_shape.Dims(1) * output_shape.Dims(2) * depth;
  const int8_t* input_data = micro::GetTensorData<int8_t>(input);
  int8_t* output_data = micro::GetTensorData<int8_t>(output);
  for (int b = 0; b < batches; ++b) {
    esp_nn_max_pool_s8(input_data + b * input_batch_size, input_shape.Dims(2),
                       input_shape.Dims(1), output_data + b * output_batch_size,
                       output_shape.Dims(2), output_shape.Dims(1),
                       params->stride_width, params->stride_height,
                       params->filter_width, params->filter_height,
                       data->padding.width, data->padding.height,
                       data->activation_min, data->activation_max, depth);
  }
  return true;
}

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPooling* data =
      static_cast<const OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32:
      AveragePoolingEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteInt8:
      AveragePoolingEvalQuantized(context, node, params, data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Input type %s is not currently supported",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPooling* data =
      static_cast<const OpDataPooling*>(node->user_data);

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32:
      MaxPoolingEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteInt8:
      if (!MaxPoolingEvalEspNn(params, data, input, output)) {
        MaxPoolingEvalQuantized(context, node, params, data, input, output);
      }
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataPooling));
}

}  // namespace

TfLiteRegistration Register_AVERAGE_POOL_2D() {
  return tflite::micro::RegisterOp(Init, PoolingPrepare, AverageEval);
}

TfLiteRegistration Register_MAX_POOL_2D() {
  return tflite::micro::RegisterOp(Init, PoolingPrepare, MaxEval);
}

}  // namespace tflite
