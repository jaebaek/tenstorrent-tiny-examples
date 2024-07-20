add_executable(tiny_tt_examples
    input_parser.cpp
    input_parser.h
    log.h
    single_tile_matmul.cpp
    single_tile_matmul.h
    blas_op.h
    buffer.cpp
    buffer.h
    utils.h
    matmul_cpu.cpp
    matmul_cpu.h
    multicast_matmul.cpp
    multicast_matmul.h
    simple_multicast.cpp
    simple_multicast.h
    main.cpp
)

target_include_directories(tiny_tt_examples PRIVATE
    ${TT_METAL_DIR}
    ${TT_METAL_DIR}/tt_metal
    ${TT_METAL_DIR}/ttnn/cpp
    ${TT_METAL_DIR}/tt_metal/common
)
target_link_libraries(tiny_tt_examples PUBLIC tt_metal m pthread)
