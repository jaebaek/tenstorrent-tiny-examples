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

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "buffer.h"
#include "conv.h"
#include "log.h"
#include "matmul_cpu.h"
#include "multicast_matmul.h"
#include "simple_multicast.h"
#include "single_tile_loopback.h"
#include "single_tile_loopback_four_cores.h"
#include "single_tile_matmul.h"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include "utils.h"

#define DEBUG 1

namespace {

template <typename T>
bool IsErrorLargerThanThreshold(std::shared_ptr<tiny::Buffer<T>> output0,
                                std::shared_ptr<tiny::Buffer<T>> output1,
                                uint32_t width, uint32_t height) {
  bool pass = true;
  auto& output_vec0 = output0->GetVector();
  auto& output_vec1 = output1->GetVector();
  uint32_t max_print_count = 0;
  for (uint32_t i = 0; i < height; ++i) {
    for (uint32_t j = 0; j < width; ++j) {
      float result0 = static_cast<float>(output_vec0[width * i + j]);
      float result1 = static_cast<float>(output_vec1[width * i + j]);
      float error = std::fabsf(result0 - result1);
      if (error > 0.006f && error > std::fabsf(result0) * 0.006f) {
#if DEBUG
        std::cout << i << ", " << j << ", " << result0 << ", " << result1
                  << std::endl;
#endif
        pass = false;
        ++max_print_count;
        if (max_print_count >= 80) return pass;
      }
    }
  }
  return pass;
}

template <>
bool IsErrorLargerThanThreshold<bfloat16>(
    std::shared_ptr<tiny::Buffer<bfloat16>> output0,
    std::shared_ptr<tiny::Buffer<bfloat16>> output1, uint32_t width,
    uint32_t height) {
  bool pass = true;
  auto& output_vec0 = output0->GetVector();
  auto& output_vec1 = output1->GetVector();
  uint32_t max_print_count = 0;
  for (uint32_t i = 0; i < height; ++i) {
    for (uint32_t j = 0; j < width; ++j) {
      float result0 = output_vec0[width * i + j].to_float();
      float result1 = output_vec1[width * i + j].to_float();
      float error = std::fabsf(result0 - result1);
      if (error > 0.025f && error > std::fabsf(result0) * 0.025f) {
#if DEBUG
        std::cout << result0 << ", " << result1 << std::endl;
#endif
        pass = false;
        ++max_print_count;
        if (max_print_count >= 80) return pass;
      }
    }
  }
  return pass;
}

template <typename T>
bool IsErrorLargerThanThreshold(std::shared_ptr<tiny::Buffer<T>> output0,
                                uint32_t from0, uint32_t to0,
                                std::shared_ptr<tiny::Buffer<T>> output1,
                                uint32_t from1, uint32_t to1) {
  assert(to0 - from0 == to1 - from1);
  bool pass = true;
  auto& output_vec0 = output0->GetVector();
  auto& output_vec1 = output1->GetVector();
  uint32_t max_print_count = 0;
  for (uint32_t i = 0; i < to1 - from1; ++i) {
    float result0 = static_cast<float>(output_vec0[i + from0]);
    float result1 = static_cast<float>(output_vec1[i + from1]);
    float error = std::fabsf(result0 - result1);
    if (error > 0.006f && error > std::fabsf(result0) * 0.006f) {
#if DEBUG
      std::cout << i << ": " << result0 << ", " << result1 << std::endl;
#endif
      pass = false;
      ++max_print_count;
      if (max_print_count >= 80) return pass;
    }
  }
  return pass;
}

template <>
bool IsErrorLargerThanThreshold<bfloat16>(
    std::shared_ptr<tiny::Buffer<bfloat16>> output0, uint32_t from0,
    uint32_t to0, std::shared_ptr<tiny::Buffer<bfloat16>> output1,
    uint32_t from1, uint32_t to1) {
  assert(to0 - from0 == to1 - from1);
  bool pass = true;
  auto& output_vec0 = output0->GetVector();
  auto& output_vec1 = output1->GetVector();
  uint32_t max_print_count = 0;
  for (uint32_t i = 0; i < to1 - from1; ++i) {
    float result0 = output_vec0[i + from0].to_float();
    float result1 = output_vec1[i + from1].to_float();
    float error = std::fabsf(result0 - result1);
    if (error > 0.025f && error > std::fabsf(result0) * 0.025f) {
#if DEBUG
      std::cout << i << ": " << result0 << ", " << result1 << std::endl;
#endif
      pass = false;
      ++max_print_count;
      if (max_print_count >= 80) return pass;
    }
  }
  return pass;
}

template <typename T>
void TestSingleTileLoopback() {
  const uint32_t number_of_elems = tiny::TileWidth() * tiny::TileHeight();
  auto input = std::make_shared<tiny::Buffer<T>>(number_of_elems, 123);
  auto output = std::make_shared<tiny::Buffer<T>>(number_of_elems);

  tiny::SingleTileLoopback<T> single_tile_loopback;
  single_tile_loopback.SetBuffers(input, output);
  single_tile_loopback.Run();

  bool pass = IsErrorLargerThanThreshold<T>(input, output, tiny::TileWidth(),
                                            tiny::TileHeight());
  if (pass) {
    log_green("-- PASS: {} --", __FUNCTION__);
  } else {
    log_error("-- FAIL: {} --", __FUNCTION__);
  }
}

template <typename T>
void TestSingleTileLoopbackFourCores() {
  const uint32_t number_of_elems = tiny::TileWidth() * tiny::TileHeight();
  auto input = std::make_shared<tiny::Buffer<T>>(number_of_elems, 123);
  auto output = std::make_shared<tiny::Buffer<T>>(4 * number_of_elems);

  tiny::SingleTileLoopbackFourCores<T> single_tile_loopback_four_cores;
  single_tile_loopback_four_cores.SetBuffers(input, output);
  single_tile_loopback_four_cores.Run();

  bool pass = IsErrorLargerThanThreshold<T>(input, 0, number_of_elems, output,
                                            0, number_of_elems);
  if (pass) log_blue("Sender output matches", __FUNCTION__);
  pass = pass &&
         IsErrorLargerThanThreshold<T>(input, 0, number_of_elems, output,
                                       number_of_elems, 2 * number_of_elems);
  if (pass) log_blue("First receiver output matches", __FUNCTION__);
  pass = pass && IsErrorLargerThanThreshold<T>(input, 0, number_of_elems,
                                               output, 2 * number_of_elems,
                                               3 * number_of_elems);
  if (pass) log_blue("Second receiver output matches", __FUNCTION__);
  pass = pass && IsErrorLargerThanThreshold<T>(input, 0, number_of_elems,
                                               output, 3 * number_of_elems,
                                               4 * number_of_elems);
  if (pass) log_blue("Third receiver output matches", __FUNCTION__);
  if (pass) {
    log_green("-- PASS: {} --", __FUNCTION__);
  } else {
    log_error("-- FAIL: {} --", __FUNCTION__);
  }
}

template <typename T>
void TestSingleTileMatrixMultiplication() {
  const uint32_t number_of_elems = tiny::TileWidth() * tiny::TileHeight();
  auto input0 = std::make_shared<tiny::Buffer<T>>(number_of_elems, 123);
  auto input1 = std::make_shared<tiny::Buffer<T>>(number_of_elems, 456);
  auto output_cpu_matmul = std::make_shared<tiny::Buffer<T>>(number_of_elems);
  auto output_single_tile_matmul =
      std::make_shared<tiny::Buffer<T>>(number_of_elems);

  tiny::CPUMatrixMultiplication<T> cpu_matmul(
      tiny::TileHeight(), tiny::TileWidth(), tiny::TileHeight());
  cpu_matmul.SetBuffers(input0, input1, output_cpu_matmul);
  cpu_matmul.Run();

  input0->Tilize(tiny::TileWidth(), tiny::TileHeight());
  input1->Tilize(tiny::TileWidth(), tiny::TileHeight());

  tiny::SingleTileMatrixMultiplication<T> single_tile_matmul;
  single_tile_matmul.SetBuffers(input0, input1, output_single_tile_matmul);
  single_tile_matmul.Run();

  output_single_tile_matmul->Untilize(tiny::TileWidth(), tiny::TileHeight());

  bool pass = IsErrorLargerThanThreshold<T>(
      output_cpu_matmul, output_single_tile_matmul, tiny::TileWidth(),
      tiny::TileHeight());
  if (pass) {
    log_green("-- PASS: {} --", __FUNCTION__);
  } else {
    log_error("-- FAIL: {} --", __FUNCTION__);
  }
}

template <typename T>
void TestSimpleMulticast() {
  const uint32_t number_of_elems = tiny::TileWidth() * tiny::TileHeight();
  auto input = std::make_shared<tiny::Buffer<T>>(number_of_elems, 123);
  auto output = std::make_shared<tiny::Buffer<T>>(4 * number_of_elems);

  tiny::SimpleMulticast<T> simple_multicast;
  simple_multicast.SetBuffers(input, output);
  simple_multicast.Run();

  bool pass = IsErrorLargerThanThreshold<T>(input, 0, number_of_elems, output,
                                            0, number_of_elems);
  if (pass) log_blue("Sender output matches", __FUNCTION__);
  pass = pass &&
         IsErrorLargerThanThreshold<T>(input, 0, number_of_elems, output,
                                       number_of_elems, 2 * number_of_elems);
  if (pass) log_blue("First receiver output matches", __FUNCTION__);
  pass = pass && IsErrorLargerThanThreshold<T>(input, 0, number_of_elems,
                                               output, 2 * number_of_elems,
                                               3 * number_of_elems);
  if (pass) log_blue("Second receiver output matches", __FUNCTION__);
  pass = pass && IsErrorLargerThanThreshold<T>(input, 0, number_of_elems,
                                               output, 3 * number_of_elems,
                                               4 * number_of_elems);
  if (pass) log_blue("Third receiver output matches", __FUNCTION__);
  if (pass) {
    log_green("-- PASS: {} --", __FUNCTION__);
  } else {
    log_error("-- FAIL: {} --", __FUNCTION__);
  }
}

template <typename T>
void TestMulticastMatrixMultiplication() {
  tt::tt_metal::Device* device = tt::tt_metal::CreateDevice(0);
  tiny::MulticastMatrixMultiplication<T> multicast_matmul(device);
  auto core_grid = device->compute_with_storage_grid_size();
  uint32_t num_cores = core_grid.x * core_grid.y;

  const uint32_t number_of_input_elems =
      num_cores * tiny::TileWidth() * tiny::TileHeight();
  auto input0 = std::make_shared<tiny::Buffer<T>>(number_of_input_elems, 123);
  auto input1 = std::make_shared<tiny::Buffer<T>>(number_of_input_elems, 456);

  const uint32_t number_of_output_elems = num_cores * number_of_input_elems;
  auto output_cpu_matmul =
      std::make_shared<tiny::Buffer<T>>(number_of_output_elems);
  auto output_multicast_matmul =
      std::make_shared<tiny::Buffer<T>>(number_of_output_elems);

  tiny::CPUMatrixMultiplication<T> cpu_matmul(num_cores * tiny::TileHeight(),
                                              tiny::TileWidth(),
                                              num_cores * tiny::TileHeight());
  cpu_matmul.SetBuffers(input0, input1, output_cpu_matmul);
  cpu_matmul.Run();

  input0->Tilize(tiny::TileWidth(), num_cores * tiny::TileHeight());
  input1->Tilize(num_cores * tiny::TileHeight(), tiny::TileWidth());

  multicast_matmul.SetBuffers(input0, input1, output_multicast_matmul);
  multicast_matmul.Run();

  output_multicast_matmul->Untilize(num_cores * tiny::TileWidth(),
                                    num_cores * tiny::TileHeight());

  bool pass = tt::tt_metal::CloseDevice(device);
  pass = pass && IsErrorLargerThanThreshold<T>(output_cpu_matmul,
                                               output_multicast_matmul,
                                               num_cores * tiny::TileWidth(),
                                               num_cores * tiny::TileHeight());

  if (pass) {
    log_green("-- PASS: {} --", __FUNCTION__);
  } else {
    log_error("-- FAIL: {} --", __FUNCTION__);
  }
}

template <typename T>
void TestConv() {
  auto input = std::make_shared<tiny::Buffer<T>>(64 * 96 * 32, 123);
  auto weight = std::make_shared<tiny::Buffer<T>>(4 * 4 * 32 * 128, 456);

  const uint32_t number_of_output_elems = 64 * 96 * 128;
  auto output_cpu_conv =
      std::make_shared<tiny::Buffer<T>>(number_of_output_elems);
  auto output_conv = std::make_shared<tiny::Buffer<T>>(number_of_output_elems);

  tiny::CpuConv<T> cpu_conv;
  cpu_conv.SetBuffers(input, weight, output_cpu_conv);
  cpu_conv.Run();
}

} /* namespace */

int main(int argc, const char* argv[]) {
  try {
    TestSingleTileLoopback<float>();
  } catch (const std::exception& e) {
    log_error("TestSingleTileLoopback::Run() failed with exception!");
    log_error("{}", e.what());
    throw;
  }

  try {
    TestSingleTileMatrixMultiplication<bfloat16>();
  } catch (const std::exception& e) {
    log_error("SingleTileMatrixMultiplication::Run() failed with exception!");
    log_error("{}", e.what());
    throw;
  }

  try {
    TestSingleTileMatrixMultiplication<float>();
  } catch (const std::exception& e) {
    log_error("SingleTileMatrixMultiplication::Run() failed with exception!");
    log_error("{}", e.what());
    throw;
  }

  // This multi-cast example is not working. Will revisit this later.
#if 0
  try {
    TestMulticastMatrixMultiplication<float>();
  } catch (const std::exception& e) {
    log_error(
        "TestMulticastMatrixMultiplication::Run() failed with exception!");
    log_error("{}", e.what());
    throw;
  }
#endif

  try {
    TestSingleTileLoopbackFourCores<float>();
  } catch (const std::exception& e) {
    log_error("TestSingleTileLoopbackFourCores::Run() failed with exception!");
    log_error("{}", e.what());
    throw;
  }

#if 1
  try {
    TestSimpleMulticast<float>();
    TestSimpleMulticast<bfloat16>();
  } catch (const std::exception& e) {
    log_error("TestSimpleMulticast::Run() for float failed with exception!");
    log_error("{}", e.what());
    throw;
  }
#endif

  TestConv<float>();
  return 0;
}
