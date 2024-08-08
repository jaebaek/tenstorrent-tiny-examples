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

#include "single_tile_matmul.h"

#include <cassert>
#include <exception>
#include <memory>
#include <tuple>

#include "buffer.h"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "utils.h"

namespace {

static const CoreCoord kSingleTileMatmulCore = {0, 0};

void _SetReaderKernel(tt::tt_metal::Program& program,
                      uint32_t input0_device_dram_address,
                      uint32_t input1_device_dram_address) {
  auto reader_id = tt::tt_metal::CreateKernel(
      program, "../../src/kernels/single_tile_matmul_reader.cpp",
      kSingleTileMatmulCore,
      tt::tt_metal::DataMovementConfig{
          .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
          .noc = NOC::RISCV_1_default});

  tt::tt_metal::SetRuntimeArgs(
      program, reader_id, kSingleTileMatmulCore,
      {input0_device_dram_address, input1_device_dram_address});
}

void _SetWriteKernel(tt::tt_metal::Program& program,
                     uint32_t output_device_dram_address) {
  auto writer_id = tt::tt_metal::CreateKernel(
      program, "../../src/kernels/single_tile_matmul_writer.cpp",
      kSingleTileMatmulCore,
      tt::tt_metal::DataMovementConfig{
          .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
          .noc = NOC::RISCV_0_default});

  tt::tt_metal::SetRuntimeArgs(program, writer_id, kSingleTileMatmulCore,
                               {output_device_dram_address});
}

void _SetComputeKernel(tt::tt_metal::Program& program) {
  tt::tt_metal::CreateKernel(
      program, "../../src/kernels/single_tile_matmul.cpp",
      kSingleTileMatmulCore,
      tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4});
}

void _SetKernels(tt::tt_metal::Program& program,
                 uint32_t input0_device_dram_address,
                 uint32_t input1_device_dram_address,
                 uint32_t output_device_dram_address) {
  _SetReaderKernel(program, input0_device_dram_address,
                   input1_device_dram_address);
  _SetWriteKernel(program, output_device_dram_address);
  _SetComputeKernel(program);
}

template <typename T>
tiny::Result _Run(std::shared_ptr<tiny::Buffer<T>> input0,
                  std::shared_ptr<tiny::Buffer<T>> input1,
                  std::shared_ptr<tiny::Buffer<T>> output) {
  input0->Tilize(tiny::TileWidth(), tiny::TileHeight());
  input1->Tilize(tiny::TileWidth(), tiny::TileHeight());

  constexpr int device_id = 0;
  tt::tt_metal::Device* device = tt::tt_metal::CreateDevice(device_id);

  tt::tt_metal::CommandQueue& command_queue = device->command_queue();
  tt::tt_metal::Program program{};

  auto input0_on_device_dram = tiny::CreateSingleTileOnDeviceDRAM<T>(device);
  auto input1_on_device_dram = tiny::CreateSingleTileOnDeviceDRAM<T>(device);
  auto output_on_device_dram = tiny::CreateSingleTileOnDeviceDRAM<T>(device);

  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_in0, program,
                                        kSingleTileMatmulCore);
  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_in1, program,
                                        kSingleTileMatmulCore);
  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_out0, program,
                                        kSingleTileMatmulCore);

  _SetKernels(program, input0_on_device_dram->address(),
              input1_on_device_dram->address(),
              output_on_device_dram->address());

  tt::tt_metal::EnqueueWriteBuffer(command_queue, input0_on_device_dram,
                                   input0->GetVector().data(), false);
  tt::tt_metal::EnqueueWriteBuffer(command_queue, input1_on_device_dram,
                                   input1->GetVector().data(), false);
  tt::tt_metal::EnqueueProgram(command_queue, program, false);
  tt::tt_metal::EnqueueReadBuffer(command_queue, output_on_device_dram,
                                  output->GetVector().data(), true);

  bool pass = tt::tt_metal::CloseDevice(device);
  return pass ? tiny::Result::kSuccess : tiny::Result::kFail;
}

} /* namespace */

namespace tiny {

template <>
Result SingleTileMatrixMultiplication<bfloat16>::Run() {
  return _Run<bfloat16>(inputs_[0], inputs_[1], output_);
}

template <>
Result SingleTileMatrixMultiplication<float>::Run() {
  return _Run<float>(inputs_[0], inputs_[1], output_);
}

} /* namespace tiny */
