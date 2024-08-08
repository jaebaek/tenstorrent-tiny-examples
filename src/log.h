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

#ifndef log_h_
#define log_h_

#include <iostream>

#include "fmt/color.h"

template <typename S, typename... Args>
inline void log_error(const S& format_str, const Args&... args) {
  std::cout << fmt::format(fmt::emphasis::bold | fg(fmt::color::red),
                           format_str, args...)
            << std::endl;
}

template <typename S, typename... Args>
inline void log_green(const S& format_str, const Args&... args) {
  std::cout << fmt::format(fmt::emphasis::bold | fg(fmt::color::green),
                           format_str, args...)
            << std::endl;
}

template <typename S, typename... Args>
inline void log_blue(const S& format_str, const Args&... args) {
  std::cout << fmt::format(fmt::emphasis::bold | fg(fmt::color::blue),
                           format_str, args...)
            << std::endl;
}

#endif /* ifndef log_h_ */
