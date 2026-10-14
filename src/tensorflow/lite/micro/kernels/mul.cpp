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

#include "tensorflow/lite/micro/kernels/mul.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "esp_nn.h"

// firewave-earth/esp-nn has esp_nn_mul_broadcast_channel_s8 from d35aac3 on,
// and every esp-nn configuration defines it as a macro.
#if !defined(esp_nn_mul_broadcast_channel_s8)
#error "MUL needs esp_nn_mul_broadcast_channel_s8 (firewave-earth/esp-nn d35aac3 or later)"
#endif

namespace tflite {

namespace {

// OpDataMul, plus whether esp_nn_mul_broadcast_channel_s8 runs the op: an int8
// MUL where one input holds one value per channel of the other (the SE scale
// [1,H,W,C] * [1,1,1,C]) or a single value. Decided once in Prepare.
struct OpDataMulEsp {
  OpDataMul mul;
  bool broadcast_channel;
  bool input1_is_small;  // input1, not input2, is the one broadcast
  int32_t spatial;       // positions of the larger input
  int32_t channels;      // values per position; 1 for a single value
};

// The number of values per position of `big` that `small` holds when MUL
// broadcasts it over all of big's other dimensions: 1 for a single value, big's
// last dimension when that is small's only dimension above 1, and 0 otherwise.
int BroadcastChannels(const RuntimeShape& big, const RuntimeShape& small) {
  if (big.DimensionsCount() == 0 || small.FlatSize() >= big.FlatSize()) {
    return 0;
  }
  if (small.FlatSize() == 1) return 1;
  const int channels = big.Dims(big.DimensionsCount() - 1);
  if (small.FlatSize() != channels) return 0;
  for (int i = 0; i + 1 < small.DimensionsCount(); ++i) {
    if (small.Dims(i) != 1) return 0;
  }
  return channels;
}

void* MulInitEsp(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataMulEsp));
}

TfLiteStatus MulPrepareEsp(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataMulEsp* data = static_cast<OpDataMulEsp*>(node->user_data);
  TF_LITE_ENSURE_OK(context,
                    CalculateOpDataMul(context, node, params, &data->mul));

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input1 =
      micro_context->AllocateTempInputTensor(node, kMulInput1Tensor);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* input2 =
      micro_context->AllocateTempInputTensor(node, kMulInput2Tensor);
  TF_LITE_ENSURE(context, input2 != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kMulOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  data->broadcast_channel = false;
  if (input1->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
    const RuntimeShape shape1 = GetTensorShape(input1);
    const RuntimeShape shape2 = GetTensorShape(input2);
    int channels = BroadcastChannels(shape1, shape2);
    bool input1_is_small = false;
    if (channels == 0) {
      channels = BroadcastChannels(shape2, shape1);
      input1_is_small = true;
    }
    const int big_size =
        input1_is_small ? shape2.FlatSize() : shape1.FlatSize();
    if (channels > 0 && GetTensorShape(output).FlatSize() == big_size) {
      data->broadcast_channel = true;
      data->input1_is_small = input1_is_small;
      data->spatial = big_size / channels;
      data->channels = channels;
    }
  }

  micro_context->DeallocateTempTfLiteTensor(input1);
  micro_context->DeallocateTempTfLiteTensor(input2);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

// esp-nn computes the same products and requantisation as BroadcastMul4DSlow,
// without its 4D index arithmetic for every element.
void EvalMulBroadcastChannelEspNn(const OpDataMulEsp* data,
                                  const TfLiteEvalTensor* input1,
                                  const TfLiteEvalTensor* input2,
                                  TfLiteEvalTensor* output) {
  const OpDataMul& mul = data->mul;
  const bool small1 = data->input1_is_small;
  esp_nn_mul_broadcast_channel_s8(
      tflite::micro::GetTensorData<int8_t>(small1 ? input2 : input1),
      tflite::micro::GetTensorData<int8_t>(small1 ? input1 : input2),
      -(small1 ? mul.input2_zero_point : mul.input1_zero_point),
      -(small1 ? mul.input1_zero_point : mul.input2_zero_point),
      tflite::micro::GetTensorData<int8_t>(output), mul.output_zero_point,
      mul.output_multiplier, mul.output_shift, mul.output_activation_min,
      mul.output_activation_max, data->spatial, data->channels);
}

}  // namespace

TfLiteStatus MulEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataMulEsp* esp_data =
      static_cast<const OpDataMulEsp*>(node->user_data);
  const OpDataMul* data = &esp_data->mul;

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kMulInput1Tensor);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kMulInput2Tensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kMulOutputTensor);

  switch (input1->type) {
    case kTfLiteInt8:
      if (esp_data->broadcast_channel) {
        EvalMulBroadcastChannelEspNn(esp_data, input1, input2, output);
      } else {
        EvalMulQuantizedReference(context, node, data, input1, input2, output);
      }
      break;
    case kTfLiteInt32:
      EvalMulQuantizedReference(context, node, data, input1, input2, output);
      break;
    case kTfLiteFloat32:
      EvalMulFloatReference(context, node, params, data, input1, input2,
                            output);
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.",
                  TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteRegistration Register_MUL() {
  return tflite::micro::RegisterOp(MulInitEsp, MulPrepareEsp, MulEval);
}

}  // namespace tflite
