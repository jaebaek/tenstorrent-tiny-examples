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

#ifndef utils_h
#define utils_h

#include <cassert>
#include <vector>

#include "tt_metal/host_api.hpp"

namespace tiny {

/* Number of elements in a row */
inline uint32_t TileWidth() { return 32; }

/* Number of elements in a column */
inline uint32_t TileHeight() { return 32; }

template <typename T>
inline uint32_t SingleTileSize() {
  return sizeof(T) * TileWidth() * TileHeight();
}

/*
 * For a given |height| by |width| matrix |buffer|, this function tilizes its
 * elements. The size of a tile on Tenstorrent Grayskull is 32x32. |height|
 * must be multiple of TileHeight() and |width| must be multiple of
 * TileWidth().
 *
 * Details:
 *
 *  |buffer| is a flatten form of all rows. In other words, when we split it
 *  into groups for every |width| elements, the first group is the first row,
 *  and the second group is the second row, and so on.
 *
 *  For example, 8x4 matrix:
 *    1 1 1 1 1 1 1 1
 *    2 2 2 2 2 2 2 2
 *    3 3 3 3 3 3 3 3
 *    4 4 4 4 4 4 4 4
 *
 *  has |buffer| like {1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,...,4}.
 *
 *  This function will split the matrix into sub-matrices (i.e., tiles) and
 *  flatten them into |buffer|.
 *
 *  If the size of tile is 4x2, the tilized form of the above example matrix
 *  will be {1,1,1,1,2,2,2,2,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,3,3,3,3,4,4,4,4}.
 *  The first 8 elements {1,1,1,1,2,2,2,2} are the left-top corner tile.
 *  The last 8 elements {3,3,3,3,4,4,4,4} are the right-bottom corner tile.
 *
 * WARNING:
 *
 *  We actually split a tile more into 4 pieces in addition to the above
 *  tilization. The hardware ISA (TT_OP_MOP) seems to require the 4
 *  sub-matrices of each tile.
 *
 * Ref:
 *  https://github.com/tenstorrent/tt-llk-gs/blob/568714a19033ad55d0f6d5a525cdb0eadaaa7e1e/common/inc/ckernel_ops.h#L286
 *  https://github.com/tenstorrent/tt-metal/blob/1b415c3487fdc59e1f8301fd44ce1d1df1f8a0bf/tt_metal/programming_examples/matmul_multi_core/matmul_multi_core.cpp
 *  https://github.com/tenstorrent/tt-metal/blob/1b415c3487fdc59e1f8301fd44ce1d1df1f8a0bf/tt_metal/common/tilize_untilize.hpp
 */
template <typename T>
void TilizeForTTDevice(std::vector<T>& buffer, uint32_t width,
                       uint32_t height) {
  assert(buffer.size() == width * height);
  assert(width % TileWidth() == 0);
  assert(height % TileHeight() == 0);

  std::vector<T> tilized_buffer(buffer.size());
  uint32_t tilized_buffer_index = 0;

  // Width and height of a sub-tile in a single tile.
  const uint32_t subTileWidth = TileWidth() / 2;
  const uint32_t subTileHeight = TileHeight() / 2;

  // Iterate tiles from the first row to the last row. Because of the tile size,
  // the outer loop interates TileHeight() rows at once.
  uint32_t elements_on_tile_row = width * TileHeight();
  for (uint32_t next_tile = 0; next_tile < buffer.size();
       next_tile += elements_on_tile_row) {
    // Access tiles from left to right. The left-top corner element index of the
    // first tile is |next_tile|.
    uint32_t right_top_corner_on_last_tile_on_row = next_tile + width;
    for (uint32_t left_top_corner_on_tile = next_tile;
         left_top_corner_on_tile < right_top_corner_on_last_tile_on_row;
         left_top_corner_on_tile += TileWidth()) {
      // Left-top sub-tile.
      for (uint32_t r = 0; r < subTileHeight; r++) {
        for (uint32_t c = 0; c < subTileWidth; c++) {
          uint32_t i = left_top_corner_on_tile + r * width + c;
          tilized_buffer[tilized_buffer_index++] = buffer[i];
        }
      }

      // Right-top sub-tile.
      for (uint32_t r = 0; r < subTileHeight; r++) {
        for (uint32_t c = subTileWidth; c < TileWidth(); c++) {
          uint32_t i = left_top_corner_on_tile + r * width + c;
          tilized_buffer[tilized_buffer_index++] = buffer[i];
        }
      }

      // Left-bottom sub-tile.
      for (uint32_t r = subTileHeight; r < TileHeight(); r++) {
        for (uint32_t c = 0; c < subTileWidth; c++) {
          uint32_t i = left_top_corner_on_tile + r * width + c;
          tilized_buffer[tilized_buffer_index++] = buffer[i];
        }
      }

      // Right-bottom sub-tile.
      for (uint32_t r = subTileHeight; r < TileHeight(); r++) {
        for (uint32_t c = subTileWidth; c < TileWidth(); c++) {
          uint32_t i = left_top_corner_on_tile + r * width + c;
          tilized_buffer[tilized_buffer_index++] = buffer[i];
        }
      }
    }
  }

  buffer = std::move(tilized_buffer);
}

template <typename T>
void UnTilizeForTTDevice(std::vector<T>& buffer, uint32_t width,
                         uint32_t height) {
  assert(buffer.size() == width * height);
  assert(width % TileWidth() == 0);
  assert(height % TileHeight() == 0);

  std::vector<T> untilized_buffer(buffer.size());
  uint32_t tilized_buffer_index = 0;

  // Width and height of a sub-tile in a single tile.
  const uint32_t subTileWidth = TileWidth() / 2;
  const uint32_t subTileHeight = TileHeight() / 2;

  // Iterate tiles from the first row to the last row. Because of the tile size,
  // the outer loop interates TileHeight() rows at once.
  uint32_t elements_on_tile_row = width * TileHeight();
  for (uint32_t next_tile = 0; next_tile < buffer.size();
       next_tile += elements_on_tile_row) {
    // Access tiles from left to right. The left-top corner element index of the
    // first tile is |next_tile|.
    uint32_t right_top_corner_on_last_tile_on_row = next_tile + width;
    for (uint32_t left_top_corner_on_tile = next_tile;
         left_top_corner_on_tile < right_top_corner_on_last_tile_on_row;
         left_top_corner_on_tile += TileWidth()) {
      // Left-top sub-tile.
      for (uint32_t r = 0; r < subTileHeight; r++) {
        for (uint32_t c = 0; c < subTileWidth; c++) {
          uint32_t i = left_top_corner_on_tile + r * width + c;
          untilized_buffer[i] = buffer[tilized_buffer_index++];
        }
      }

      // Right-top sub-tile.
      for (uint32_t r = 0; r < subTileHeight; r++) {
        for (uint32_t c = subTileWidth; c < TileWidth(); c++) {
          uint32_t i = left_top_corner_on_tile + r * width + c;
          untilized_buffer[i] = buffer[tilized_buffer_index++];
        }
      }

      // Left-bottom sub-tile.
      for (uint32_t r = subTileHeight; r < TileHeight(); r++) {
        for (uint32_t c = 0; c < subTileWidth; c++) {
          uint32_t i = left_top_corner_on_tile + r * width + c;
          untilized_buffer[i] = buffer[tilized_buffer_index++];
        }
      }

      // Right-bottom sub-tile.
      for (uint32_t r = subTileHeight; r < TileHeight(); r++) {
        for (uint32_t c = subTileWidth; c < TileWidth(); c++) {
          uint32_t i = left_top_corner_on_tile + r * width + c;
          untilized_buffer[i] = buffer[tilized_buffer_index++];
        }
      }
    }
  }

  buffer = std::move(untilized_buffer);
}

template <typename T>
tt::DataFormat GetDataFormat() {
  if (typeid(T) == typeid(bfloat16)) {
    return tt::DataFormat::Float16_b;
  } else if (typeid(T) == typeid(float)) {
    return tt::DataFormat::Float32;
  } else if (typeid(T) == typeid(int)) {
    return tt::DataFormat::Int32;
  }
  return tt::DataFormat::Invalid;
}

template <typename T>
std::shared_ptr<tt::tt_metal::Buffer> CreateBufferOnDeviceDRAM(
    tt::tt_metal::Device* device, uint32_t size_in_bytes) {
  tt::tt_metal::InterleavedBufferConfig device_dram_conf{
      .device = device,
      .size = size_in_bytes,
      .page_size = tiny::SingleTileSize<T>(),
      .buffer_type = tt::tt_metal::BufferType::DRAM};
  return std::move(CreateBuffer(device_dram_conf));
}

template <typename T>
std::shared_ptr<tt::tt_metal::Buffer> CreateSingleTileOnDeviceDRAM(
    tt::tt_metal::Device* device) {
  return std::move(
      CreateBufferOnDeviceDRAM<T>(device, tiny::SingleTileSize<T>()));
}

template <typename T>
void CreateCircularBufferOnDevice(uint32_t circular_buffer_id,
                                  tt::tt_metal::Program& program,
                                  CoreRange cores,
                                  uint32_t number_of_tiles = 1) {
  tt::DataFormat format = tiny::GetDataFormat<T>();
  assert(format != tt::DataFormat::Invalid);

  tt::tt_metal::CircularBufferConfig conf(
      number_of_tiles * tiny::SingleTileSize<T>(),
      {{circular_buffer_id, format}});
  conf = conf.set_page_size(circular_buffer_id, tiny::SingleTileSize<T>());
  tt::tt_metal::CreateCircularBuffer(program, cores, conf);
}

} /* namespace tiny */

#endif /* ifndef utils_h */
