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

#include "simple_multicast.h"

#include <variant>

#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "utils.h"

namespace {

static const CoreRange kAllCores = {{0, 0}, {0, 3}};
static const CoreCoord kSenderCore = {0, 0};
static const CoreRange kReceiverCores = {{0, 1}, {0, 3}};

void _SetDataMoveKernel(tt::tt_metal::Program& program,
                        uint32_t input_device_dram_address,
                        uint32_t receiver_sema_addr,
                        uint32_t output_device_dram_address) {
  auto sender_id = tt::tt_metal::CreateKernel(
      program,
      "../../src/3_simple_multicast/kernels/simple_multicast_sender_reader.cpp",
      kSenderCore,
      tt::tt_metal::DataMovementConfig{
          .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
          .noc = NOC::RISCV_1_default});

  tt::tt_metal::SetRuntimeArgs(program, sender_id, kSenderCore,
                               {input_device_dram_address, receiver_sema_addr,
                                output_device_dram_address});

  auto receiver_id = tt::tt_metal::CreateKernel(
      program,
      "../../src/3_simple_multicast/kernels/"
      "simple_multicast_receiver_reader.cpp",
      kReceiverCores,
      tt::tt_metal::DataMovementConfig{
          .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
          .noc = NOC::RISCV_1_default});

  for (uint32_t i = 1; i < 4; ++i) {
    CoreCoord core = {0, i};
    tt::tt_metal::SetRuntimeArgs(
        program, receiver_id, core,
        {i, receiver_sema_addr, output_device_dram_address});
  }
}

template <typename T>
tiny::Result _Run(std::shared_ptr<tiny::Buffer<T>> input,
                  std::shared_ptr<tiny::Buffer<T>> output) {
  tt::tt_metal::Device* device = tt::tt_metal::CreateDevice(0);

  tt::tt_metal::CommandQueue& command_queue = device->command_queue();
  tt::tt_metal::Program program{};

  auto input_on_device_dram =
      tiny::CreateBufferOnDeviceDRAM<T>(device, input->GetSizeInBytes());
  auto output_on_device_dram =
      tiny::CreateBufferOnDeviceDRAM<T>(device, output->GetSizeInBytes());

  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_in0, program, kAllCores);

  auto receiver_sema_addr =
      tt::tt_metal::CreateSemaphore(program, kAllCores, 0);

  _SetDataMoveKernel(program, input_on_device_dram->address(),
                     receiver_sema_addr, output_on_device_dram->address());

  tt::tt_metal::EnqueueWriteBuffer(command_queue, input_on_device_dram,
                                   input->GetVector().data(), false);
  tt::tt_metal::EnqueueProgram(command_queue, program, false);
  tt::tt_metal::EnqueueReadBuffer(command_queue, output_on_device_dram,
                                  output->GetVector().data(), true);

  bool pass = tt::tt_metal::CloseDevice(device);
  return pass ? tiny::Result::kSuccess : tiny::Result::kFail;
}

} /* namespace */

namespace tiny {

template <>
Result SimpleMulticast<bfloat16>::Run() {
  return _Run<bfloat16>(input_, output_);
}

template <>
Result SimpleMulticast<float>::Run() {
  return _Run<float>(input_, output_);
}

} /* namespace tiny */
