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

#include "input_parser.h"

#include <cstring>
#include <iostream>
#include <string>
#include <tuple>

#include "fmt/color.h"
#include "log.h"

static void PrintUsage(const char* program_path, bool print_description) {
  std::cout << "Usage:" << std::endl;
  std::cout << "\t" << program_path << " <M> <K> <N>" << std::endl << std::endl;

  if (!print_description) {
    std::cout << "Use --help option for more information." << std::endl;
    return;
  }

  std::cout << "A program to compare matrix multiplications running on CPU and "
            << "TT device." << std::endl;
  std::cout << "When dimensions of two matrices are M by K and K by N, you "
            << "must provide M, K, and N." << std::endl;
  std::cout << "This program will generate random 16-bits float values for "
            << "two matrices and multiply" << std::endl;
  std::cout << "them on CPU and TT device." << std::endl;
  std::cout << "This program will compare the results and output pass or fail."
            << std::endl
            << std::endl;
}

std::tuple<InputParsingResult, uint32_t, uint32_t, uint32_t> ParseInputs(
    int argc, char** argv) {
  if (argc == 1) {
    log_error("{} needs more arguments!!", argv[0]);
    PrintUsage(argv[0], false);
    return std::make_tuple(INVALID_INPUTS, 0, 0, 0);
  }

  if (argc == 2) {
    if (strncmp(argv[1], "--help", 6) == 0 && argv[1][6] == '\0') {
      PrintUsage(argv[0], true);
      return std::make_tuple(HELP_OPTION, 0, 0, 0);
    }
    log_error("Invalid argument!!");
    PrintUsage(argv[0], false);
    return std::make_tuple(INVALID_INPUTS, 0, 0, 0);
  }

  if (argc > 2 && argc != 4) {
    log_error("Invalid number of arguments!!");
    PrintUsage(argv[0], false);
    return std::make_tuple(INVALID_INPUTS, 0, 0, 0);
  }

  try {
    int m = std::stoi(argv[1]);
    int k = std::stoi(argv[2]);
    int n = std::stoi(argv[3]);

    if (m < 0 || k < 0 || n < 0) {
      log_error("Invalid negative matrix dimension!!");
      PrintUsage(argv[0], false);
      return std::make_tuple(INVALID_INPUTS, 0, 0, 0);
    }
    return std::make_tuple(VALID_INPUTS, m, k, n);
  } catch (const std::exception& e) {
    log_error("Invalid arguments: {}", e.what());
    PrintUsage(argv[0], false);
    return std::make_tuple(INVALID_INPUTS, 0, 0, 0);
  }

  // Unreachable
  return std::make_tuple(INVALID_INPUTS, 0, 0, 0);
}
