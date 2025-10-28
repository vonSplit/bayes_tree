cmake_minimum_required(VERSION 3.22)
project(BayesTree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create library bayes_tree from the source files
add_library(bayes_tree
    src/bayes_tree.cpp
    src/categorical_distribution.cpp
    src/node.cpp
)

# target_include_directories(bayes_tree PUBLIC include) # Old skool


## Target properties (modern CMake: target-based settings)
target_compile_features(bayes_tree PUBLIC cxx_std_20)
set_target_properties(bayes_tree PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(bayes_tree
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Create test executable test_tree from the test source file
add_executable(test_tree tests/test_tree.cpp)
# test_tree has access to bayes_tree library
target_link_libraries(test_tree PRIVATE bayes_tree)

add_executable(test_categorical_distribution tests/test_categorical_distribution.cpp)
target_link_libraries(test_categorical_distribution PRIVATE bayes_tree)

# Enable CTest support  
enable_testing()
# Register tests with CTest: NAME <test_name> COMMAND <executable>
add_test(NAME test_tree COMMAND test_tree)
add_test(NAME test_categorical_distribution COMMAND test_categorical_distribution)


# Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if (BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()



