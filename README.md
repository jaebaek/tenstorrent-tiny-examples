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
$ clang-format -style=file -i [modified-files]
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

`src/single_tile_matmul.cpp` implements a matrix multiplication between two tiles.
It uses a single Tensix core `{0, 0}`. `src/kernels/single_tile_matmul_reader.cpp`
reads two input matrices from device DRAM and passes them to `CB::c_in0` and
`CB::c_in1` buffers that are on L1 cache.

On its compute kernel `src/kernels/single_tile_matmul.cpp`, it uses
`matmul_tiles(..)` that conducts `DST = A * B` like
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

# About license

* At the moment I am writing this document, I am working for Google.
Based on Google's open-source policy, I open-sourced this project with
Apache 2.0 license, but this is my personal side project and this
project is not related to any project I am working on at Google.
