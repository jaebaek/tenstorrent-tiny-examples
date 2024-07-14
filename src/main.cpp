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
#include "log.h"
#include "matmul_cpu.h"
#include "multicast_matmul.h"
#include "single_tile_matmul.h"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include "utils.h"

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
        std::cout << i << ", " << j << ", " << result0 << ", " << result1
                  << std::endl;
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
        std::cout << result0 << ", " << result1 << std::endl;
        pass = false;
        ++max_print_count;
        if (max_print_count >= 80) return pass;
      }
    }
  }
  return pass;
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
void TestMulticastMatrixMultiplication() {
  tiny::MulticastMatrixMultiplication<T> multicast_matmul;
  auto device = multicast_matmul.GetOrCreateDevice();
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

//  output_multicast_matmul->Untilize(num_cores * tiny::TileWidth(),
//                                    num_cores * tiny::TileHeight());

  bool pass = IsErrorLargerThanThreshold<T>(
      output_cpu_matmul, output_multicast_matmul, num_cores * tiny::TileWidth(),
      num_cores * tiny::TileHeight());

  // -------
  multicast_matmul.CloseDevice();

  const uint32_t number_of_elems = tiny::TileWidth() * tiny::TileHeight();

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "output_multicast_matmul:" << std::endl;
  for (uint32_t i = 0; i < num_cores; ++i) {
    std::cout << output_multicast_matmul->GetVector()[i * number_of_elems] << std::endl;
    std::cout << output_multicast_matmul->GetVector()[i * number_of_elems + 1] << std::endl;
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  auto i0 = std::make_shared<tiny::Buffer<T>>(number_of_elems);
  auto i1 = std::make_shared<tiny::Buffer<T>>(number_of_elems);
  input0->Untilize(tiny::TileWidth(), num_cores * tiny::TileHeight());
  input1->Untilize(num_cores * tiny::TileHeight(), tiny::TileWidth());
  for (uint32_t i = 0; i < number_of_elems; ++i) {
    i0->GetVector()[i] = input0->GetVector()[i];
    i1->GetVector()[i] = input1->GetVector()[i];
  }
  auto output_single_tile_matmul =
      std::make_shared<tiny::Buffer<T>>(number_of_elems);
  tiny::SingleTileMatrixMultiplication<T> single_tile_matmul;
  single_tile_matmul.SetBuffers(i0, i1, output_single_tile_matmul);
  single_tile_matmul.Run();
  output_single_tile_matmul->Untilize(tiny::TileWidth(), tiny::TileHeight());
  std::cout << "single matmul:" << std::endl;
  for (uint32_t i = 0; i < 80; ++i) {
    std::cout << output_single_tile_matmul->GetVector()[i] << std::endl;
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  for (uint32_t i = 0; i < 32; ++i) {
    std::cout << input0->GetVector()[i] << std::endl;
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  for (uint32_t i = 0; i < num_cores; ++i) {
    std::cout << input0->GetVector()[i * number_of_elems] << std::endl;
    std::cout << input0->GetVector()[i * number_of_elems + 1] << std::endl;
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  for (uint32_t i = 0; i < num_cores; ++i) {
    std::cout << input1->GetVector()[i * tiny::TileWidth()] << std::endl;
    std::cout << input1->GetVector()[i * tiny::TileWidth() + 1] << std::endl;
  }
  // -------

  if (pass) {
    log_green("-- PASS: {} --", __FUNCTION__);
  } else {
    log_error("-- FAIL: {} --", __FUNCTION__);
  }
}

} /* namespace */

int main(int argc, const char* argv[]) {
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

  try {
    TestMulticastMatrixMultiplication<float>();
  } catch (const std::exception& e) {
    log_error(
        "TestMulticastMatrixMultiplication::Run() failed with exception!");
    log_error("{}", e.what());
    throw;
  }

  return 0;
}
