cmake_minimum_required(VERSION 3.20)

project(flow4d_dfwavelet LANGUAGES CXX)

add_library(flow4d_dfwavelet SHARED
    dfwavelet.cpp
    dfwavelet_bridge.cpp
)

target_compile_features(flow4d_dfwavelet PRIVATE cxx_std_17)
target_include_directories(flow4d_dfwavelet PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(flow4d_dfwavelet PROPERTIES
    CXX_EXTENSIONS OFF
    OUTPUT_NAME "flow4d_dfwavelet"
    PREFIX ""
)

if(MSVC)
    target_compile_options(flow4d_dfwavelet PRIVATE /O2 /W4)
else()
    target_compile_options(flow4d_dfwavelet PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()
