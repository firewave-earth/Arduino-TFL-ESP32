/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "esp_nn.h"
#include <esp_heap_caps.h>

#include <cstring>

namespace tflite {
namespace {

// The single-channel kernel below is plain C, and the library builds with -Os,
// which would keep it from inlining the requantisation or unrolling the taps.
#pragma GCC push_options
#pragma GCC optimize("O3")

// Filters with more taps than this keep the esp-nn path.
constexpr int kMaxSingleChannelTaps = 25;

// MultiplyByQuantizedMultiplier written out so it inlines.
// SaturatingRoundingDoublingHighMul's sign-dependent nudge and truncating
// division equal a constant nudge and a floor shift by 31, and it cannot
// saturate because the multiplier is never INT32_MIN. The rounding shift is
// RoundingDivideByPOT's own 32-bit form: a variable 64-bit shift would be a
// libgcc call on Xtensa.
inline int32_t RequantizeChannel(int32_t x, int32_t multiplier, int shift) {
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  const int64_t product = static_cast<int64_t>(x * (1 << left_shift)) * multiplier;
  const int32_t high = static_cast<int32_t>((product + (int64_t{1} << 30)) >> 31);
  const int32_t mask = static_cast<int32_t>((uint32_t{1} << right_shift) - 1);
  const int32_t remainder = high & mask;
  const int32_t threshold = (mask >> 1) + (high < 0);
  return (high >> right_shift) + (remainder > threshold);
}

// int8 CONV_2D with one input channel, such as the PaSST student's stem (3x3,
// 128x400x1 to 48 channels). esp-nn pads each 3-byte filter row to 16 SIMD lanes
// there, so most lanes multiply zeros; this gathers each pixel's window once and
// reuses it for every output channel. Same arithmetic as
// reference_integer_ops::ConvPerChannel: taps outside the image contribute
// nothing, then bias, requantisation, offset and clamp in the same order.
// kTaps is the filter's tap count when known at compile time, 0 otherwise.
template <int kTaps>
void ConvSingleInputChannelInt8(const ConvParams& params,
                                const int32_t* output_multiplier,
                                const int32_t* output_shift,
                                const RuntimeShape& input_shape,
                                const int8_t* input_data,
                                const RuntimeShape& filter_shape,
                                const int8_t* filter_data,
                                const int32_t* bias_data,
                                const RuntimeShape& output_shape,
                                int8_t* output_data) {
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int taps = kTaps > 0 ? kTaps : filter_height * filter_width;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);

  int32_t window[kMaxSingleChannelTaps];
  int8_t* out = output_data;
  for (int batch = 0; batch < batches; ++batch) {
    const int8_t* image = input_data + batch * input_height * input_width;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin =
          out_y * params.stride_height - params.padding_values.height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin =
            out_x * params.stride_width - params.padding_values.width;
        // Input plus offset at every tap, 0 where the tap is outside the image.
        int k = 0;
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + params.dilation_height_factor * filter_y;
          for (int filter_x = 0; filter_x < filter_width; ++filter_x, ++k) {
            const int in_x =
                in_x_origin + params.dilation_width_factor * filter_x;
            const bool inside = in_x >= 0 && in_x < input_width && in_y >= 0 &&
                                in_y < input_height;
            window[k] =
                inside ? image[in_y * input_width + in_x] + params.input_offset
                       : 0;
          }
        }
        const int8_t* filter = filter_data;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          int32_t acc = 0;
          for (int t = 0; t < taps; ++t) acc += filter[t] * window[t];
          filter += taps;
          if (bias_data) acc += bias_data[out_channel];
          acc = RequantizeChannel(acc, output_multiplier[out_channel],
                                  output_shift[out_channel]);
          acc += params.output_offset;
          acc = std::max(acc, params.quantized_activation_min);
          acc = std::min(acc, params.quantized_activation_max);
          *out++ = static_cast<int8_t>(acc);
        }
      }
    }
  }
}

#if defined(ARCH_ESP32_S3)
// One esp-nn window: the input under each tap, so up to 16 taps.
constexpr int kWindowBytes = 16;
// Output pixels per esp_nn_conv_s8_window16 call: the windows stay small in
// internal RAM, and the strided bytes each call writes stay in the data cache
// until the other channels fill in their lines.
constexpr int kWindowPixels = 128;

// The same convolution as ConvSingleInputChannelInt8 on the S3's SIMD unit: each
// pixel's taps are gathered into a 16-byte window, then esp-nn takes one output
// channel at a time across the row, one vector multiply-accumulate per output.
// Returns false, having written nothing, when esp-nn cannot reproduce the
// reference exactly for these quantisation parameters or the scratch buffer
// cannot be allocated.
bool ConvSingleInputChannelWindow16(const ConvParams& params,
                                    const int32_t* output_multiplier,
                                    const int32_t* output_shift,
                                    const RuntimeShape& input_shape,
                                    const int8_t* input_data,
                                    const RuntimeShape& filter_shape,
                                    const int8_t* filter_data,
                                    const int32_t* bias_data,
                                    const RuntimeShape& output_shape,
                                    int8_t* output_data) {
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int taps = filter_height * filter_width;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  // Taps outside the image read the input zero point, so they add nothing once
  // the input offset is applied.
  const int32_t fill = -params.input_offset;
  if (taps > kWindowBytes || fill < -128 || fill > 127) return false;

  // One window past the last pixel: the kernel loads ahead.
  const size_t window_bytes = (kWindowPixels + 1) * kWindowBytes;
  const size_t filter_bytes = output_depth * kWindowBytes;
  const size_t quant_bytes = output_depth * sizeof(conv_window16_quant_t);
  auto* scratch = static_cast<int8_t*>(heap_caps_aligned_alloc(
      16, window_bytes + filter_bytes + quant_bytes,
      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (!scratch) return false;
  int8_t* windows = scratch;
  int8_t* filters = scratch + window_bytes;
  auto* quant = reinterpret_cast<conv_window16_quant_t*>(filters + filter_bytes);

  // Lanes past the last tap stay 0; the filters are 0 there too.
  memset(scratch, 0, window_bytes + filter_bytes);
  for (int c = 0; c < output_depth; ++c) {
    int8_t* filter = filters + c * kWindowBytes;
    memcpy(filter, filter_data + c * taps, taps);
    if (esp_nn_conv_s8_window16_quant_init(
            filter, bias_data ? bias_data[c] : 0, params.input_offset,
            params.output_offset, output_multiplier[c], output_shift[c],
            params.quantized_activation_min, params.quantized_activation_max,
            &quant[c]) != 0) {
      heap_caps_free(scratch);
      return false;
    }
  }

  for (int batch = 0; batch < batches; ++batch) {
    const int8_t* image = input_data + batch * input_height * input_width;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin =
          out_y * params.stride_height - params.padding_values.height;
      int8_t* out_row =
          output_data +
          ((batch * output_height + out_y) * output_width) * output_depth;
      for (int x0 = 0; x0 < output_width; x0 += kWindowPixels) {
        const int n = std::min(kWindowPixels, output_width - x0);
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + params.dilation_height_factor * filter_y;
          const int8_t* row = in_y >= 0 && in_y < input_height
                                  ? image + in_y * input_width
                                  : nullptr;
          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
            int8_t* lane = windows + filter_y * filter_width + filter_x;
            // Pixel i reads row[(x0 + i) * stride + offset]; [first, last)
            // are the pixels where that is inside the image.
            const int offset = params.dilation_width_factor * filter_x -
                               params.padding_values.width;
            int first = 0, last = row ? n : 0;
            while (first < last &&
                   (x0 + first) * params.stride_width + offset < 0) {
              ++first;
            }
            while (last > first &&
                   (x0 + last - 1) * params.stride_width + offset >=
                       input_width) {
              --last;
            }
            for (int i = 0; i < first; ++i) lane[i * kWindowBytes] = fill;
            if (first < last) {
              const int8_t* in = row + (x0 + first) * params.stride_width + offset;
              for (int i = first; i < last; ++i, in += params.stride_width) {
                lane[i * kWindowBytes] = *in;
              }
            }
            for (int i = last; i < n; ++i) lane[i * kWindowBytes] = fill;
          }
        }
        int8_t* out = out_row + x0 * output_depth;
        for (int c = 0; c < output_depth; ++c) {
          esp_nn_conv_s8_window16(windows, filters + c * kWindowBytes, out + c, n,
                                  output_depth, &quant[c]);
        }
      }
    }
  }
  heap_caps_free(scratch);
  return true;
}
#endif  // ARCH_ESP32_S3

#pragma GCC pop_options

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(
      context,
      input->type == filter->type ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8),
      "Hybrid models are not supported on TFLite Micro.");

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data), tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt16: {
      switch (bias->type) {
        case kTfLiteInt32: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<std::int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        case kTfLiteInt64: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<std::int64_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        default:
          MicroPrintf("Bias type %s (%d) not supported.",
                      TfLiteTypeGetName(bias->type), bias->type);
          return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt8: {
      const auto& input_shape = tflite::micro::GetTensorShape(input);
      const auto& filter_shape = tflite::micro::GetTensorShape(filter);
      const auto& output_shape = tflite::micro::GetTensorShape(output);
      const auto conv_params = ConvParamsQuantized(params, data);

      const int taps = filter_shape.Dims(1) * filter_shape.Dims(2);
      if (input_shape.Dims(3) == 1 && filter_shape.Dims(3) == 1 &&
          taps <= kMaxSingleChannelTaps) {
#if defined(ARCH_ESP32_S3)
        if (ConvSingleInputChannelWindow16(
                conv_params, data.per_channel_output_multiplier,
                data.per_channel_output_shift, input_shape,
                tflite::micro::GetTensorData<int8_t>(input), filter_shape,
                tflite::micro::GetTensorData<int8_t>(filter),
                tflite::micro::GetOptionalTensorData<int32_t>(bias),
                output_shape, tflite::micro::GetTensorData<int8_t>(output))) {
          break;
        }
#endif
        auto* conv = taps == 9 ? ConvSingleInputChannelInt8<9>
                               : ConvSingleInputChannelInt8<0>;
        conv(conv_params, data.per_channel_output_multiplier,
             data.per_channel_output_shift, input_shape,
             tflite::micro::GetTensorData<int8_t>(input), filter_shape,
             tflite::micro::GetTensorData<int8_t>(filter),
             tflite::micro::GetOptionalTensorData<int32_t>(bias), output_shape,
             tflite::micro::GetTensorData<int8_t>(output));
        break;
      }

      data_dims_t input_dims = {input_shape.Dims(2), input_shape.Dims(1),
                                input_shape.Dims(3), 0};
      data_dims_t filter_dims = {filter_shape.Dims(2), filter_shape.Dims(1),
                                 filter_shape.Dims(3), filter_shape.Dims(0)};
      data_dims_t output_dims = {output_shape.Dims(2), output_shape.Dims(1),
                                 output_shape.Dims(3), 0};
      conv_params_t esp_conv_params = {};
      esp_conv_params.in_offset = conv_params.input_offset;
      esp_conv_params.out_offset = conv_params.output_offset;
      esp_conv_params.stride = {conv_params.stride_width, conv_params.stride_height};
      esp_conv_params.padding = {conv_params.padding_values.width,
                                 conv_params.padding_values.height};
      esp_conv_params.dilation = {conv_params.dilation_width_factor,
                                  conv_params.dilation_height_factor};
      esp_conv_params.activation = {conv_params.quantized_activation_min,
                                    conv_params.quantized_activation_max};
      quant_data_t quant = {data.per_channel_output_shift,
                            data.per_channel_output_multiplier};

      int scratch_size = esp_nn_get_conv_scratch_size(
          &input_dims, &filter_dims, &output_dims, &esp_conv_params);
      void* scratch = nullptr;
      if (scratch_size > 0) {
        scratch = heap_caps_malloc(scratch_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!scratch) scratch = heap_caps_malloc(scratch_size, MALLOC_CAP_8BIT);
        if (scratch) esp_nn_set_conv_scratch_buf(scratch);
      }

      esp_nn_conv_s8(&input_dims,
                     tflite::micro::GetTensorData<int8_t>(input),
                     &filter_dims,
                     tflite::micro::GetTensorData<int8_t>(filter),
                     tflite::micro::GetOptionalTensorData<int32_t>(bias),
                     &output_dims,
                     tflite::micro::GetTensorData<int8_t>(output),
                     &esp_conv_params, &quant);
      if (scratch) heap_caps_free(scratch);
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(Init, ConvPrepare, Eval);
}

}  // namespace tflite
