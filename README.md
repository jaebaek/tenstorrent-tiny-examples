# Simple Tenstorrent tt-metalium examples with explanation

* Target hardware: GraySkull e75

### Build

* Install software stack for TT-Metalium: https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md
  * Follow step 1 to 4.

```
$ cd tenstorrent-tiny-examples
$ git submodule update --init --recursive
$ export ARCH_NAME=grayskull
$ export TT_METAL_HOME=$PWD/external/tt-metal
$ mkdir build
$ cd build
$ cmake -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17 -GNinja ..
$ ninja install
```

### Run

```
$ cd conv-example-tt-grayskull
$ export ARCH_NAME=grayskull
$ export TT_METAL_HOME=$PWD/external/tt-metal
$ ./bin/tiny_tt_examples
```

### Code format

```
$ find ./src/ -name "*.h" -or -name "*.cpp" -exec  clang-format -style=file -i {} \;
```

# Memo (about important concepts)

### `bfloat16`

IEEE Standard for Floating-Point Arithmetic (IEEE 754) specifies that a 32-bits float consists of:
* 1 bit for sign.
* 8 bits for exponent.
* 23 bits for fraction (mantissa).

`bfloat16` consists of:
* 1 bit for sign.
* 8 bits for exponent.
* 7 bits for fraction (mantissa).

tt-metalium keeps `bfloat16` in a unsigned 16-bits integer. We can convert a float to `bfloat16`
by right-shifting it (simply dropping the lower 16-bits of the float value). We can convert
`bfloat16` to a float by left-shifting it (adding lower 16-bits of zeros).

### Tilization

#### Basic concept of tiling for matrix multiplication

When we run a matrix multiplication `A * B` where `A` and `B` are matrices, we have to run
dot-projects of a row of `A` with all columns of `B`. If we naively implement it, our program
can access each row of `A` as many as the number of columns of `B`. Ideally, we want to keep
a row of `A` in registers (or fast data storage like L1 cache), but we will experience the
shortage of the fast data storage (register spilling issue).

"Tiling" is a technique to access a sub-matrix (we call it "tile") of `A` and a sub-matrix
of `B` for the multiplication. In that way, we can keep all elements of the tiles into the
limited fast data storage, and we can also conduct all multiplications between elements
on the tiles. By sliding tiles, we can continue.

When we have multiple layers of cache like GPU, we can utilize more storage classes depending
on thread groups. See [this blog post from Nvidia](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
for details.

Other references are:
* https://github.com/NVIDIA/cutlass
* https://github.com/google/uVkCompute

#### Tilization on TT device

A Tensix core on a Tenstorrent device has a [matrix engine](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md).
It supports the matrix multiplication whose dimension is 32 by 32. Tenstorrent named
this unit matrix "tile". I guess the term is from the "tiling" of matrix multiplication.

Note that Tenstorrent documents call a group of tiles "block" when the "block" is the size
of "tile" in the above tiling technique.

The tilization means reorder the elements of an array or vector like `std::vector`
to provide the data in the order of tiles to Tensix cores.
For example, for a simpler explanation, let me assume a dimension of a tile is
4 by 4, and we want to tilize a matrix `A` whose values are
```
1 1 1 1 5 5 5 5
2 2 2 2 6 6 6 6
3 3 3 3 7 7 7 7
4 4 4 4 8 8 8 8
```
that is kept by a `std::vector<int>` like `{1,1,1,1,5,5,5,5,2,2,2,2,6,6,6,6,...}`.
We have two tiles
```
1 1 1 1
2 2 2 2
3 3 3 3
4 4 4 4
```
and
```
5 5 5 5
6 6 6 6
7 7 7 7
8 8 8 8
```
so after tilizing it must be `{1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,...}`.

The only difference of the tilization on Tenstorrent device from the above example
is the dimension of the tile matrix. As explained above, it is 32 by 32.

The untilization is the opposite conversion.

### Matrix multiplication

`src/4_single_tile_matmul/single_tile_matmul.cpp` implements a matrix
multiplication between two tiles.  It uses a single Tensix core `{0, 0}`.
`src/4_single_tile_matmul/kernels/single_tile_matmul_reader.cpp`
reads two input matrices from device DRAM and passes them to `CB::c_in0` and
`CB::c_in1` buffers that are on L1 cache.

On its compute kernel `src/4_single_tile_matmul/kernels/single_tile_matmul.cpp`,
it uses `matmul_tiles(..)` that conducts `DST = A * B` like
`DST = tile on CB::c_in0 * tile on CB::c_in1`.

Note that it uses `DST`, so we have to `acquire_dst(..)` and `release_dst(..)`.
When I asked why we need to "acquire" `DST` register, Tenstorrent folks answered
that we have 3 RISC-V processors on a single Tensix core, and they can cause a
data race (access to `DST` in a random order).

Also note that **I guess** `matmul_tiles(..)` conducts `DST += A * B` from [this
example](https://github.com/tenstorrent/tt-metal/blob/7b883eb5b4bb225bfdf2edcf253d3dc7cfbbd400/tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp#L36-L42). It is a compute kernel to multiply
two large matrices with multiple tiles on each row/column. In that case, for each
output tile, we have to slide all tiles on the corresponding row of `A` and all
tiles on the corresponding column of `B`, and we have to add all the mat-mul
results between their tiles. In the kernel, we do not have "add" operation, but
it works. That's why I concluded that `matmul_tiles(..)` conducts `DST += A * B`
instead of `DST = A * B`.

### Multi-cast (WIP)

I investigated the [multi-cast example provided by Tenstorrent](
https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp)
in depth, and I implemented my own example to test it.
My plan for `src/multicast_matmul.cpp` is:

* Create two matrices that have dimensions `number of cores * tile height (32)`
  by `tile width` and `tile width` by `number of cores * tile height (32)`.
* Each i-th core reads i-th row tile of the first matrix (`A`) and i-th column
  of the second matrix (`B`).
* The reader kernel iterates `i` from `[0, number of cores)`. When `i` is the
  core itself (I gave core id to all of them from `[0, number of cores)`), it
  becomes "sender" of the multicast. Otherwise, it becomes "receiver".
  * "sender" sends the tile it reads for the i-th column of `B` to all other
    cores.
  * Each compute kernel runs the mat-mul for two tiles for the i-th row of `A`
    and the i-th column of `B`.

Currently, the mat-mul result is different between CPU and my example, but
the multi-cast seems to be working. I printed each first float value of tile
from receivers and senders. When I run `./bin/tiny_tt_examples` twice after
setting `export TT_METAL_DPRINT_CORES=0,0` and
`export TT_METAL_DPRINT_CORES=3,7` (or print from other cores), respectively,
the results showed that the first value of matrices between receivers and
senders are matching.

### Compute kernel debugging tip: UNPACK, MATH, PACK kernels

A compute kernel looks like a single kernel, but `tt::tt_metal::CreateKernel(..)`
actually generates three kernels: UNPACK, MATH, PACK kernels.

You can check it with the following steps:
* For a simple TT program, intentionally add a syntax error to the compute kernel.
* Running the program will fail with compile errors like:
```
cd /path/to/conv-example-tt-grayskull/external/tt-metal//built/2052/kernels/simple_multicast/1584599061800683236/trisc2/ && /path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-g++ -mgrayskull -march=rv32iy -mtune=rvtt-b1 -mabi=ilp32 -std=c++17 -flto -ffast-math -fno-use-cxa-atexit -fno-exceptions -Wall -Werror -Wno-unknown-pragmas -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -Wno-unused-function -O3 -DARCH_GRAYSKULL -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 -DDEBUG_PRINT_ENABLED -DUCK_CHLKC_PACK -DNAMESPACE=chlkc_pack -DCOMPILE_FOR_TRISC=2 -DKERNEL_BUILD -I. -I.. -I/path/to/conv-example-tt-grayskull/external/tt-metal// -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/include -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/inc -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/inc/debug -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/inc/grayskull -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/inc/grayskull/grayskull_defines -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/inc/grayskull/noc -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/third_party/umd/device/grayskull -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/ckernels/grayskull/metal/common -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/ckernels/grayskull/metal/llk_io -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/third_party/tt_llk_grayskull/common/inc -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/third_party/tt_llk_grayskull/llk_lib -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/ckernels/grayskull/inc -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/ckernels/grayskull/metal/common -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/ckernels/grayskull/metal/llk_io -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/ckernels/grayskull/metal/llk_api -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/ckernels/grayskull/metal/llk_api/llk_sfpu -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/third_party/sfpi/include -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/firmware/src -I/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/third_party/tt_llk_grayskull/llk_lib -c -o trisck.o /path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/hw/firmware/src/trisck.cc
```
* If you check the directory e.g.,
`/path/to/conv-example-tt-grayskull/external/tt-metal//built/2052/kernels/simple_multicast/1584599061800683236/trisc2/`
you will see the build artifact. For example, `trisc2/trisc2.elf` is an ELF
binary for RISC-V. We can run
`/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-objdump`
and
`/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-readelf`
to investigate the binary.
* We can use `-E` option for the above
`/path/to/conv-example-tt-grayskull/external/tt-metal//tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-g++`
command to get the preprocess result.
  * Unpack kernel option: `-DUCK_CHLKC_UNPACK -DNAMESPACE=chlkc_unpack`
  * Math kernel option: `-DUCK_CHLKC_MATH -DNAMESPACE=chlkc_math`
  * Pack kernel option: `-DUCK_CHLKC_PACK -DNAMESPACE=chlkc_pack`

Interestingly, for the following kernel (simple\_multicast kernel):
```
static inline SliceRange hw_all() {
  return SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
}

namespace NAMESPACE {
void MAIN {
  acquire_dst(tt::DstMode::Tile);

  cb_wait_front(tt::CB::c_in0, /* number of tiles */ 1);
  copy_tile_to_dst_init_short();
  copy_tile(tt::CB::c_in0, 0, /* DST */ 0);
#if TINY_DEBUG
  DPRINT_UNPACK(DPRINT << TSLICE(tt::CB::c_in0, 0, hw_all()) << ENDL());
#endif
  cb_pop_front(tt::CB::c_in0, /* number of tiles */ 1);

  // PACK(( llk_pack_hw_configure_disaggregated<false,
  // DST_ACCUM_MODE>(tt::CB::c_out0) ));
  PACK((llk_pack_init(tt::CB::c_out0)));
  // PACK(( llk_setup_outputs()  ));
  PACK((llk_pack_dest_init<false, DST_ACCUM_MODE>(tt::CB::c_out0)));

  LOG(DPRINT << "[COMPUTE] pack tile" << ENDL());

  cb_reserve_back(tt::CB::c_out0, /* number of tiles */ 1);
  pack_tile(/* DST */ 0, tt::CB::c_out0);
  cb_push_back(tt::CB::c_out0, /* number of tiles */ 1);

#if TINY_DEBUG
  DPRINT_PACK(DPRINT << TSLICE(tt::CB::c_out0, 0, hw_all()) << ENDL());
#endif

  release_dst(tt::DstMode::Tile);
  LOG(DPRINT << "[COMPUTE] done" << ENDL());
}
}  // namespace NAMESPACE
```

The PACK kernel uses only the packing part of APIs. For example,
`cb_wait_front(..)` is not related to packing. The preprocessed
`cb_wait_front(..)` for PACK kernel is empty:
```
inline __attribute__((always_inline)) void cb_wait_front(uint32_t cbid, uint32_t ntiles) {
    ;
}
```

On the other hand, the UNPACK kernel has the following
`cb_wait_front(..)`:
```
inline __attribute__((always_inline)) void cb_wait_front(uint32_t cbid, uint32_t ntiles) {
    ( llk_wait_tiles(cbid, ntiles) );
}
```

To implement this different preprocessed results, tt-metal uses
`PACK(..), MATH(..), UNPACK(..)` macro.

Another interesting part is `acquire_dst(tt::DstMode mode)`.
The UNPACK kernel has an empty one:
```
inline __attribute__((always_inline)) void acquire_dst(tt::DstMode mode) {
    ;

    ;
}
```
The MATH kernel waits for DEST available:
```
inline __attribute__((always_inline)) void acquire_dst(tt::DstMode mode) {
    ( llk_math_wait_for_dest_available() );

    ;
}
```
The UNPACK kernel waits for the end of MATH kernel:
```
inline __attribute__((always_inline)) void acquire_dst(tt::DstMode mode) {
    ;

    ( llk_packer_wait_for_math_done() );
}
```

[Its implementation](https://github.com/tenstorrent/tt-metal/blob/6d4951a20ca4c392888f924f038ae0780a8cc656/tt_metal/include/compute_kernel_api/reg_api.h#L28-L32) matches the preprocessed code:
```
ALWI void acquire_dst(tt::DstMode mode) {
    MATH(( llk_math_wait_for_dest_available()  ));

    PACK(( llk_packer_wait_for_math_done()  ));
}
```

Based on the implementation of `acquire_dst(..)`, if we use it,
we can guess it executes UNPACK, MATH, PACK in order.

# About license

* At the moment I am writing this document, I am working for Google.
Based on Google's open-source policy, I open-sourced this project with
Apache 2.0 license, but this is my personal side project and this
project is not related to any project I am working on at Google.
