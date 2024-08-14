add_executable(tiny_tt_examples
    input_parser.cpp
    input_parser.h
    log.h
    blas_op.h
    buffer.cpp
    buffer.h
    utils.cpp
    utils.h
    matmul_cpu.cpp
    matmul_cpu.h
    multicast_matmul.cpp
    multicast_matmul.h
    1_single_tile_loopback/single_tile_loopback.cpp
    1_single_tile_loopback/single_tile_loopback.h
    2_single_tile_loopback_four_cores/single_tile_loopback_four_cores.cpp
    2_single_tile_loopback_four_cores/single_tile_loopback_four_cores.h
    3_simple_multicast/simple_multicast.cpp
    3_simple_multicast/simple_multicast.h
    4_single_tile_matmul/single_tile_matmul.cpp
    4_single_tile_matmul/single_tile_matmul.h
    5_multicast_advanced/multicast_advanced.cpp
    5_multicast_advanced/multicast_advanced.h
    conv.cpp
    conv.h
    main.cpp
)

target_include_directories(tiny_tt_examples PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TT_METAL_DIR}
    ${TT_METAL_DIR}/tt_metal
    ${TT_METAL_DIR}/ttnn/cpp
    ${TT_METAL_DIR}/tt_metal/common
)
target_link_libraries(tiny_tt_examples PUBLIC tt_metal m pthread)
