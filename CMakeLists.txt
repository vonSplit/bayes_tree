cmake_minimum_required(VERSION 3.22)
project(BayesTree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library
add_library(bayes_tree
    src/bayes_tree.cpp
    src/node.cpp
)
## Target properties (modern CMake: target-based settings)
target_compile_features(bayes_tree PUBLIC cxx_std_20)
set_target_properties(bayes_tree PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(bayes_tree
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Test executable
add_executable(test_tree tests/test_tree.cpp)
target_link_libraries(test_tree PRIVATE bayes_tree)

# Enable CTest
enable_testing()
add_test(NAME test_tree COMMAND test_tree)
