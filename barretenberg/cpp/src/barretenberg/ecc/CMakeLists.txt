barretenberg_module(ecc numeric crypto_keccak crypto_sha256)

if(DISABLE_ADX)
    message(STATUS "Disabling ADX assembly variant.")
    target_compile_definitions(ecc PRIVATE -DDISABLE_ADX)
endif()

# Conditionally add CUDA support
if(ENABLE_CUDA)
    target_sources(ecc PRIVATE ../gpu/msm_cuda.cu)
    target_link_libraries(ecc PRIVATE ${CUDA_LIBRARIES})
endif()

target_precompile_headers(
    ecc
    PUBLIC
    "$<$<COMPILE_LANGUAGE:CXX>:<random$<ANGLE-R>>"
    $<$<COMPILE_LANGUAGE:CXX>:"barretenberg/numeric/uint256/uint256.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"barretenberg/numeric/uint256/uint256_impl.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/asm_macros.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/field_declarations.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/field_impl.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/field_impl_generic.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/field_impl_x64.hpp">
    $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CURRENT_SOURCE_DIR}/fields/field.hpp">
)
