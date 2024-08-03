// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file was initially copied from tt-metal github repo and
// updated by Jaebaek Seo.

// Copyright (c) 2024 Jaebaek Seo.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "conv.h"

#include <algorithm>
#include <cassert>

#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"
#include "utils.h"

namespace tiny {
namespace {

constexpr std::uint32_t input_h = 64;
constexpr std::uint32_t input_w = 96;
constexpr std::uint32_t input_c = 32;
constexpr std::uint32_t weight_h = 4;
constexpr std::uint32_t weight_w = 4;
constexpr std::uint32_t slide_h = 1;
constexpr std::uint32_t slide_w = 1;
constexpr std::uint32_t padding_h = weight_h / 2;
constexpr std::uint32_t padding_w = weight_w / 2;
constexpr std::uint32_t output_h =
    (input_h + 2 * padding_h - weight_h) / slide_h;
constexpr std::uint32_t output_w =
    (input_w + 2 * padding_w - weight_w) / slide_w;
constexpr std::uint32_t output_c = 128;

inline void CheckInputDimension(uint32_t number_of_elems) {
  assert(number_of_elems == input_h * input_w * input_c);
}

inline void CheckWeightDimension(uint32_t number_of_elems) {
  assert(number_of_elems == weight_h * weight_w * input_c * output_c);
}

inline void CheckOutputDimension(uint32_t number_of_elems) {
  assert(number_of_elems == output_h * output_w * output_c);
}

template <typename T>
tiny::Result RunTT(std::shared_ptr<tiny::Buffer<T>> input,
                   std::shared_ptr<tiny::Buffer<T>> weight,
                   std::shared_ptr<tiny::Buffer<T>> output) {
  /*
  input->Tilize(tiny::TileWidth(), num_cores * tiny::TileHeight());
  input1->Tilize(num_cores * tiny::TileHeight(), tiny::TileWidth());
  output_multicast_matmul->Untilize(num_cores * tiny::TileWidth(),
                                    num_cores * tiny::TileHeight());
  */

  // return pass ? tiny::Result::kSuccess : tiny::Result::kFail;
  return tiny::Result::kSuccess;
}

template <typename T>
inline float accumulate(float a, T b, T c) {
  return a + b * c;
}

template <>
inline float accumulate<bfloat16>(float a, bfloat16 b, bfloat16 c) {
  return a + b.to_float() * c.to_float();
}

template <typename T>
inline T convert_to_type(float a) {
  return static_cast<T>(a);
}

template <>
inline bfloat16 convert_to_type<bfloat16>(float a) {
  return bfloat16(a);
}

template <typename T>
tiny::Result RunCpu(std::shared_ptr<tiny::Buffer<T>> input_buffer,
                    std::shared_ptr<tiny::Buffer<T>> weight_buffer,
                    std::shared_ptr<tiny::Buffer<T>> output_buffer) {
  auto& input = input_buffer->GetVector();
  auto& weight = weight_buffer->GetVector();
  auto& output = output_buffer->GetVector();

  // oc = output channel
  for (uint32_t oc = 0; oc < output_c; ++oc) {
    for (uint32_t i = 0; i < output_h; ++i) {
      for (uint32_t j = 0; j < output_w; ++j) {
        float output_element = 0;

        // ic = input channel
        for (uint32_t ic = 0; ic < input_c; ++ic) {
          // Assuming that padding_h = weight_h / 2 and padding_w = weight_w / 2
          for (uint32_t row = i - padding_h; row < i + padding_h; ++row) {
            for (uint32_t col = j - padding_w; col < j + padding_w; ++col) {
              T input_value = 0;
              if (0 <= row && row < output_h && 0 <= col && col < output_w) {
                input_value =
                    input[ic * input_h * input_w + row * input_w + col];
              }
              int wr = row - i + padding_h;
              int wc = col - j + padding_w;
              T weight_value =
                  weight[ic * weight_h * weight_w + wr * weight_w + wc];

              output_element =
                  accumulate<T>(output_element, input_value, weight_value);
            }
          }
        }

        output[oc * output_h * output_w + i * output_w + j] =
            convert_to_type<T>(output_element);
      }
    }
  }
  return tiny::Result::kSuccess;
}

} /* namespace  */

template <>
Result Conv<bfloat16>::Run() {
  return RunTT<bfloat16>(input_, weight_, output_);
}

template <>
Result Conv<float>::Run() {
  return RunTT<float>(input_, weight_, output_);
}

template <>
Result CpuConv<bfloat16>::Run() {
  return RunCpu<bfloat16>(input_, weight_, output_);
}

template <>
Result CpuConv<float>::Run() {
  return RunCpu<float>(input_, weight_, output_);
}

template <>
void Conv<bfloat16>::CheckDimension() {
  CheckInputDimension(input_->GetNumberOfElements());
  CheckWeightDimension(weight_->GetNumberOfElements());
  CheckOutputDimension(output_->GetNumberOfElements());
}

template <>
void Conv<float>::CheckDimension() {
  CheckInputDimension(input_->GetNumberOfElements());
  CheckWeightDimension(weight_->GetNumberOfElements());
  CheckOutputDimension(output_->GetNumberOfElements());
}

} /* namespace tiny */

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

void matmul_multicore_reuse_mcast(vector<bfloat16>& a, vector<bfloat16>& b,
                                  vector<bfloat16>& output, bool bcast_batch,
                                  uint32_t M, uint32_t N, uint32_t K,
                                  uint32_t B, Device* device) {
  /*
   * Setup program to execute along with its buffers and kernels to use
   * Core range is just single core
   */
  CommandQueue& cq = device->command_queue();
  Program program{};

  tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
  MathFidelity math_fidelity = MathFidelity::HiFi4;
  uint32_t single_tile_size = detail::TileSize(cb_data_format);
  // uint32_t single_tile_size = 2 * 1024;

  auto compute_with_storage_grid_size =
      device->compute_with_storage_grid_size();
  uint32_t num_cores_x = compute_with_storage_grid_size.x;
  uint32_t num_cores_y = compute_with_storage_grid_size.y;

  /*
   * EXtracting Matrix dimensions from input/output vectors
   */
  // C = A*B
  // MN = MK*KN
  uint32_t Mt = M / TILE_HEIGHT;
  uint32_t Kt = K / TILE_WIDTH;
  uint32_t Nt = N / TILE_WIDTH;
  uint32_t KtNt = Kt * Nt;
  uint32_t MtKt = Mt * Kt;
  uint32_t MtNt = Mt * Nt;

  // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie.
  // multiples of 16*32 x 16*32) NOTE: Maximum number of tiles in output is 120
  // * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])2
  uint32_t in0_block_w = 2;
  // uint32_t out_subblock_h = 4;
  // uint32_t out_subblock_w = 2;
  // uint32_t per_core_M = 16;
  // uint32_t per_core_N = 16;

  // Get large matmul params
  auto matmul_params = bmm_op_utils::get_large_matmul_params(
      Mt, Nt, num_cores_y, num_cores_x, in0_block_w);

  // How many tiles among output columns will be handled by a core.
  uint32_t per_core_M = std::get<0>(matmul_params);

  // How many tiles among output rows will be handled by a core.
  uint32_t per_core_N = std::get<1>(matmul_params);

  // Number of tiles in a column of a subblock.
  // For example, if a column has 8 * 8 * 32 elements, and a column has 8
  // subblocks, |out_subblock_h| will be (8 * 8 * 32) / 8 / 32 = 8 tiles.
  uint32_t out_subblock_h = std::get<2>(matmul_params);

  // Number of tiles in a row of a subblock.
  uint32_t out_subblock_w = std::get<3>(matmul_params);

  log_info(tt::LogVerif, " -- Metalium Core Sizing --");
  log_info(tt::LogVerif,
           " -- per_core_M= {} -- per_core_N= {} -- out_subblock_h= {} -- "
           "out_subblock_w= {} --",
           per_core_M, per_core_N, out_subblock_h, out_subblock_w);

  TT_ASSERT(Mt % per_core_M == 0);
  TT_ASSERT(Nt % per_core_N == 0);
  TT_ASSERT(Kt % in0_block_w == 0);

  uint32_t in0_block_tiles = per_core_M * in0_block_w;
  uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
  uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
  uint32_t in1_block_tiles = per_core_N * in0_block_w;
  uint32_t in1_CB_tiles = in1_block_tiles * 2;  // double buffer
  uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
  uint32_t out_block_tiles = per_core_M * per_core_N;
  uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
  uint32_t out_CB_size = out_CB_tiles * single_tile_size;

  // Compute kernel compile time args
  uint32_t num_blocks = (Kt / in0_block_w);

  uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
  uint32_t in0_block_num_tiles =
      out_subblock_h * in0_block_w * in0_num_subblocks;
  uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

  uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
  uint32_t in1_block_num_tiles =
      out_subblock_w * in0_block_w * in1_num_subblocks;
  uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

  uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

  vector<uint32_t> compute_kernel_args = {
      in0_block_w,             // in0_block_w
      in0_num_subblocks,       // in0_num_subblocks
      in0_block_num_tiles,     // in0_block_num_tiles
      in0_subblock_num_tiles,  // in0_subblock_num_tiles

      in1_num_subblocks,    // in1_num_subblocks
      in1_block_num_tiles,  // in1_block_num_tiles
      in1_per_core_w,       // in1_per_core_w

      num_blocks,  // num_blocks

      out_subblock_h,          // out_subblock_h
      out_subblock_w,          // out_subblock_w
      out_subblock_num_tiles,  // out_subblock_num_tiles
      B                        // batch
  };

  /*
   * Multi-Core prep
   */
  uint32_t num_blocks_y = Mt / per_core_M;
  uint32_t num_blocks_x = Nt / per_core_N;
  uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
  TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
  CoreCoord start_core = {0, 0};
  CoreCoord core_range = bmm_op_utils::get_core_range(
      num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);

  uint32_t start_core_x = start_core.x;
  uint32_t start_core_y = start_core.y;
  uint32_t num_cores_c = core_range.x;
  uint32_t num_cores_r = core_range.y;

  CoreRange all_cores({(std::size_t)start_core_x, (std::size_t)start_core_y},
                      {(std::size_t)start_core_x + num_cores_c - 1,
                       (std::size_t)start_core_y + num_cores_r - 1});

  CoreRange left_column(
      {(std::size_t)start_core_x, (std::size_t)start_core_y},
      {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

  CoreRange all_except_left_column(
      {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
      {(std::size_t)start_core_x + num_cores_c - 1,
       (std::size_t)start_core_y + num_cores_r - 1});

  CoreRange in0_sender_in1_sender(
      {(std::size_t)start_core_x, (std::size_t)start_core_y},
      {(std::size_t)start_core_x, (std::size_t)start_core_y});

  CoreRange in0_sender_in1_receiver(
      {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
      {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});

  CoreRange in0_receiver_in1_sender(
      {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
      {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});

  CoreRange in0_receiver_in1_receiver(
      {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
      {(std::size_t)start_core_x + num_cores_c - 1,
       (std::size_t)start_core_y + num_cores_r - 1});

  //////////////////////////////////////////////////
  /*
   * Create DRAM Buffers for input and output vectors
   * Writing data from input vectors to source buffers
   */

  uint32_t dram_buffer_A_size =
      single_tile_size * Mt *
      Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
  uint32_t dram_buffer_B_size =
      single_tile_size * Nt *
      Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
  uint32_t dram_buffer_C_size =
      single_tile_size * Mt *
      Nt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
  tt_metal::InterleavedBufferConfig dram_config_A{
      .device = device,
      .size = dram_buffer_A_size,
      .page_size = single_tile_size,
      .buffer_type = tt_metal::BufferType::DRAM};

  tt_metal::InterleavedBufferConfig dram_config_B{
      .device = device,
      .size = dram_buffer_B_size,
      .page_size = single_tile_size,
      .buffer_type = tt_metal::BufferType::DRAM};

  tt_metal::InterleavedBufferConfig dram_config_C{
      .device = device,
      .size = dram_buffer_B_size,
      .page_size = single_tile_size,
      .buffer_type = tt_metal::BufferType::DRAM};

  auto src0_dram_buffer = CreateBuffer(dram_config_A);
  auto src1_dram_buffer = CreateBuffer(dram_config_B);
  auto dst_dram_buffer = CreateBuffer(dram_config_C);
  uint32_t src0_addr = src0_dram_buffer->address();
  uint32_t src1_addr = src1_dram_buffer->address();
  uint32_t dst_addr = dst_dram_buffer->address();

  /*
   * Config of Circular Buffer in the device L1
   * input tiles count is = 2 because it's single tile process, and
   * double-buffer
   */
  uint32_t src0_cb_index = CB::c_in0;  // 0
  CircularBufferConfig cb_src0_config =
      CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
          .set_page_size(src0_cb_index, single_tile_size);
  auto cb_src0 =
      tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

  uint32_t src1_cb_index = CB::c_in1;  // 1
  CircularBufferConfig cb_src1_config =
      CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
          .set_page_size(src1_cb_index, single_tile_size);
  auto cb_src1 =
      tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

  uint32_t output_cb_index = CB::c_out0;  // output operands start at index 16
  uint32_t interm0_cb_index = 24;
  std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
      {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
  CircularBufferConfig cb_output_config =
      CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
          .set_page_size(output_cb_index, single_tile_size)
          .set_page_size(interm0_cb_index, single_tile_size);
  auto cb_output = tt_metal::CreateCircularBuffer(
      program, CoreRangeSet({all_cores}), cb_output_config);

  ////////////////////////////
  /*
   * Compile time arguments
   */
  bool src0_is_dram =
      src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
  bool src1_is_dram =
      src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
  std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram,
                                                    (uint32_t)src1_is_dram};

  bool dst_is_dram =
      dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
  // std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)
  // output_cb_index, (uint32_t)dst_is_dram};
  std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

  /*
   * Create Kernels (Reader, Writer, Compute)
   */
  // Create reader and writer kernels per core group

  auto mm_reader_kernel_in0_sender_in1_sender_id = tt_metal::CreateKernel(
      program,
      "tt_metal/programming_examples/matmul_common/kernels/dataflow/"
      "reader_bmm_tile_layout_in0_sender_in1_sender.cpp",
      in0_sender_in1_sender,
      tt_metal::DataMovementConfig{
          .processor = tt_metal::DataMovementProcessor::RISCV_1,
          .noc = tt_metal::NOC::RISCV_0_default,
          .compile_args = reader_compile_time_args});

  auto mm_reader_kernel_in0_sender_in1_receiver_id = tt_metal::CreateKernel(
      program,
      "tt_metal/programming_examples/matmul_common/kernels/dataflow/"
      "reader_bmm_tile_layout_in0_sender_in1_receiver.cpp",
      in0_sender_in1_receiver,
      tt_metal::DataMovementConfig{
          .processor = tt_metal::DataMovementProcessor::RISCV_1,
          .noc = tt_metal::NOC::RISCV_0_default,
          .compile_args = reader_compile_time_args});

  auto mm_reader_kernel_in0_receiver_in1_sender_id = tt_metal::CreateKernel(
      program,
      "tt_metal/programming_examples/matmul_common/kernels/dataflow/"
      "reader_bmm_tile_layout_in0_receiver_in1_sender.cpp",
      in0_receiver_in1_sender,
      tt_metal::DataMovementConfig{
          .processor = tt_metal::DataMovementProcessor::RISCV_1,
          .noc = tt_metal::NOC::RISCV_1_default,
          .compile_args = reader_compile_time_args});

  auto mm_reader_kernel_in0_receiver_in1_receiver_id = tt_metal::CreateKernel(
      program,
      "tt_metal/programming_examples/matmul_common/kernels/dataflow/"
      "reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp",
      in0_receiver_in1_receiver,
      tt_metal::DataMovementConfig{
          .processor = tt_metal::DataMovementProcessor::RISCV_1,
          .noc = tt_metal::NOC::RISCV_1_default,
          .compile_args = reader_compile_time_args});

  auto unary_writer_kernel_noc0_id = tt_metal::CreateKernel(
      program,
      "tt_metal/programming_examples/matmul_common/kernels/dataflow/"
      "writer_bmm_tile_layout.cpp",
      all_except_left_column,
      tt_metal::DataMovementConfig{
          .processor = tt_metal::DataMovementProcessor::RISCV_0,
          .noc = tt_metal::NOC::RISCV_0_default,
          .compile_args = writer_compile_time_args});

  auto unary_writer_kernel_noc1_id = tt_metal::CreateKernel(
      program,
      "tt_metal/programming_examples/matmul_common/kernels/dataflow/"
      "writer_bmm_tile_layout.cpp",
      left_column,
      tt_metal::DataMovementConfig{
          .processor = tt_metal::DataMovementProcessor::RISCV_0,
          .noc = tt_metal::NOC::RISCV_1_default,
          .compile_args = writer_compile_time_args});

  // Create compute kernel
  auto mm_kernel_id = tt_metal::CreateKernel(
      program,
      "tt_metal/programming_examples/matmul_common/kernels/compute/"
      "bmm_large_block_zm.cpp",
      all_cores,
      tt_metal::ComputeConfig{.math_fidelity = math_fidelity,
                              .compile_args = compute_kernel_args});

  auto in0_mcast_sender_semaphore =
      tt_metal::CreateSemaphore(program, all_cores, INVALID);
  auto in0_mcast_receiver_semaphore =
      tt_metal::CreateSemaphore(program, all_cores, INVALID);
  auto in1_mcast_sender_semaphore =
      tt_metal::CreateSemaphore(program, all_cores, INVALID);
  auto in1_mcast_receiver_semaphore =
      tt_metal::CreateSemaphore(program, all_cores, INVALID);

  /*
   * Kernels - Runtime arguments
   */
  std::vector<KernelHandle> reader_kernel_ids;
  std::vector<KernelHandle> writer_kernel_ids;
  for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
    for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
      CoreCoord core = {(std::size_t)start_core_x + core_idx_x,
                        (std::size_t)start_core_y + core_idx_y};

      CoreCoord left_core = {(std::size_t)start_core_x, (std::size_t)core.y};
      CoreCoord left_core_plus_one = {(std::size_t)start_core_x + 1,
                                      (std::size_t)core.y};
      CoreCoord right_core = {(std::size_t)start_core_x + num_cores_c - 1,
                              (std::size_t)core.y};
      CoreCoord top_core = {(std::size_t)core.x, (std::size_t)start_core_y};
      CoreCoord top_core_plus_one = {(std::size_t)core.x,
                                     (std::size_t)start_core_y + 1};
      CoreCoord bottom_core = {(std::size_t)core.x,
                               (std::size_t)start_core_y + num_cores_r - 1};

      auto left_core_physical =
          device->worker_core_from_logical_core(left_core);
      auto left_core_plus_one_physical =
          device->worker_core_from_logical_core(left_core_plus_one);
      auto right_core_physical =
          device->worker_core_from_logical_core(right_core);
      auto top_core_physical = device->worker_core_from_logical_core(top_core);
      auto top_core_plus_one_physical =
          device->worker_core_from_logical_core(top_core_plus_one);
      auto bottom_core_physical =
          device->worker_core_from_logical_core(bottom_core);

      std::vector<uint32_t> mm_reader_args = {
          (std::uint32_t)src0_dram_buffer->address(),  // in0_buffer_addr
          (std::uint32_t)Kt * per_core_M *
              core_idx_y,              // in0_buffer_start_tile_id
          (std::uint32_t)1,            // in0_buffer_stride_w
          (std::uint32_t)Kt,           // in0_buffer_stride_h
          (std::uint32_t)in0_block_w,  // in0_buffer_next_block_stride

          (std::uint32_t)in0_block_w,               // in0_block_w
          (std::uint32_t)per_core_M,                // in0_block_h
          (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles

          (std::uint32_t)src1_dram_buffer->address(),  // in1_buffer_addr
          (std::uint32_t)per_core_N * core_idx_x,  // in1_buffer_start_tile_id
          (std::uint32_t)1,                        // in1_buffer_stride_w
          (std::uint32_t)Nt,                       // in1_buffer_stride_h
          (std::uint32_t)in0_block_w * Nt,  // in1_buffer_next_block_stride

          (std::uint32_t)per_core_N,                // in1_block_w
          (std::uint32_t)in0_block_w,               // in1_block_h
          (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles

          (std::uint32_t)Kt / in0_block_w,  // num_blocks

          (std::uint32_t)right_core_physical.x,  // in0_mcast_dest_noc_start_x
          (std::uint32_t)right_core_physical.y,  // in0_mcast_dest_noc_start_y
          (std::uint32_t)
              left_core_plus_one_physical.x,  // in0_mcast_dest_noc_end_x
          (std::uint32_t)
              left_core_plus_one_physical.y,    // in0_mcast_dest_noc_end_y
          (std::uint32_t)(num_cores_c - 1),     // in0_mcast_num_dests
          (std::uint32_t)left_core_physical.x,  // in0_mcast_sender_noc_x
          (std::uint32_t)left_core_physical.y,  // in0_mcast_sender_noc_y
          (std::uint32_t)in0_mcast_sender_semaphore,
          (std::uint32_t)in0_mcast_receiver_semaphore,

          (std::uint32_t)bottom_core_physical.x,  // in0_mcast_dest_noc_start_x
          (std::uint32_t)bottom_core_physical.y,  // in0_mcast_dest_noc_start_y
          (std::uint32_t)
              top_core_plus_one_physical.x,  // in0_mcast_dest_noc_end_x
          (std::uint32_t)
              top_core_plus_one_physical.y,    // in0_mcast_dest_noc_end_y
          (std::uint32_t)(num_cores_r - 1),    // in0_mcast_num_dests
          (std::uint32_t)top_core_physical.x,  // in0_mcast_sender_noc_x
          (std::uint32_t)top_core_physical.y,  // in0_mcast_sender_noc_y
          (std::uint32_t)in1_mcast_sender_semaphore,
          (std::uint32_t)in1_mcast_receiver_semaphore,

          (std::uint32_t)Mt * Kt,     // MtKt
          (std::uint32_t)Kt * Nt,     // KtNt
          (std::uint32_t)B,           // batch
          (std::uint32_t)bcast_batch  // bcast_B
      };

      std::vector<uint32_t> writer_args = {
          (std::uint32_t)dst_dram_buffer->address(),  // out_buffer_addr
          (std::uint32_t)core_idx_x * per_core_N +
              core_idx_y * per_core_M * Nt,  // out_buffer_start_tile_id
          (std::uint32_t)1,                  // out_buffer_stride_w
          (std::uint32_t)Nt,                 // out_buffer_stride_h
          (std::uint32_t)out_subblock_w,  // out_buffer_next_subblock_stride_w
          (std::uint32_t)out_subblock_h *
              Nt,  // out_buffer_next_subblock_stride_h

          (std::uint32_t)out_subblock_w,  // out_subblock_w
          (std::uint32_t)out_subblock_h,  // out_subblock_h
          (std::uint32_t)(out_subblock_w *
                          out_subblock_h),  // out_subblocks_w * out_subblocks_h
          (std::uint32_t)(per_core_N / out_subblock_w),  // out_num_subblocks_w
          (std::uint32_t)(per_core_M / out_subblock_h),  // out_num_subblocks_h

          (std::uint32_t)Mt * Nt,  // MtNt
          (std::uint32_t)B         // batch
      };

      if (core_idx_x == 0 and core_idx_y == 0) {
        tt_metal::SetRuntimeArgs(program,
                                 mm_reader_kernel_in0_sender_in1_sender_id,
                                 core, mm_reader_args);  // RISCV_0_default
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core,
                                 writer_args);  // RISCV_1_default
        reader_kernel_ids.push_back(mm_reader_kernel_in0_sender_in1_sender_id);
        writer_kernel_ids.push_back(unary_writer_kernel_noc1_id);
      } else if (core_idx_x == 0 and core_idx_y != 0) {
        tt_metal::SetRuntimeArgs(program,
                                 mm_reader_kernel_in0_sender_in1_receiver_id,
                                 core, mm_reader_args);  // RISCV_0_default
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core,
                                 writer_args);  // RISCV_1_default
        reader_kernel_ids.push_back(
            mm_reader_kernel_in0_sender_in1_receiver_id);
        writer_kernel_ids.push_back(unary_writer_kernel_noc1_id);
      } else if (core_idx_x != 0 and core_idx_y == 0) {
        tt_metal::SetRuntimeArgs(program,
                                 mm_reader_kernel_in0_receiver_in1_sender_id,
                                 core, mm_reader_args);  // RISCV_1_default
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core,
                                 writer_args);  // RISCV_0_default
        reader_kernel_ids.push_back(
            mm_reader_kernel_in0_receiver_in1_sender_id);
        writer_kernel_ids.push_back(unary_writer_kernel_noc0_id);
      } else {
        tt_metal::SetRuntimeArgs(program,
                                 mm_reader_kernel_in0_receiver_in1_receiver_id,
                                 core, mm_reader_args);  // RISCV_1_default
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core,
                                 writer_args);  // RISCV_0_default
        reader_kernel_ids.push_back(
            mm_reader_kernel_in0_receiver_in1_receiver_id);
        writer_kernel_ids.push_back(unary_writer_kernel_noc0_id);
      }
    }
  }

  /* Launch program & read in output buffer result into the host vector */

  EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
  EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
  EnqueueProgram(cq, program, false);
  EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
}

///////////////////////////////////////

int run(int argc, const char* argv[]) {
  bool pass = true;

  if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
    TT_THROW("Test not supported w/ slow dispatch, exiting");
  }

  try {
    /* Silicon accelerator setup */
    constexpr int device_id = 0;
    Device* device = CreateDevice(device_id);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie.
    // multiples of 16*32 x 16*32) NOTE: Maximum number of tiles in output is
    // 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])

    /* Create source data */
    constexpr uint32_t M = 3200;  // user-defined
    constexpr uint32_t N = 3200;  // user-defined
    constexpr uint32_t K = 3200;  // user-defined
    constexpr uint32_t B = 1;     // user-defined

    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    constexpr uint32_t single_tile_size = 2 * 1024;
    uint32_t dram_buffer_A_size =
        single_tile_size * Mt * Kt;  // num_tiles of FP16_B
    uint32_t dram_buffer_B_size =
        single_tile_size * Nt * Kt;  // num_tiles of FP16_B
    uint32_t dram_buffer_C_size =
        single_tile_size * Mt * Nt;  // num_tiles of FP16_B

    /* input vectors */
    std::vector<bfloat16> src0_vec = create_random_vector_of_bfloat16_native(
        dram_buffer_A_size, 1, 123, -0.4);
    std::vector<bfloat16> src1_vec = create_random_vector_of_bfloat16_native(
        dram_buffer_B_size, 1, 12522, -0.3);

    /* Input vector tilizing */
    tiny::TilizeForTTDevice(src0_vec, M, K);
    tiny::TilizeForTTDevice(src1_vec, K, N);

    /* Calling the MatMul host program. Read in result into a host vector */
    vector<bfloat16> result_vec(dram_buffer_C_size / sizeof(bfloat16));
    matmul_multicore_reuse_mcast(src0_vec, src1_vec, result_vec, false, M, N, K,
                                 B, device);
    tiny::UnTilizeForTTDevice(result_vec, M, N);

    log_info(tt::LogVerif, "Output vector of size {}", result_vec.size());

    pass &= CloseDevice(device);

  } catch (const std::exception& e) {
    tt::log_error(tt::LogTest, "Test failed with exception!");
    tt::log_error(tt::LogTest, "{}", e.what());

    throw;
  }

  if (pass) {
    tt::log_info(tt::LogTest, "Test Passed");
  } else {
    TT_THROW("Test Failed");
  }

  TT_ASSERT(pass);

  return 0;
}
