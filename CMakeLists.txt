cmake_minimum_required(VERSION 3.22)
project(BayesTree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create library bayes_tree from the source files
add_library(bayes_tree
    src/bayes_tree.cpp
    src/categorical_distribution.cpp
    src/dirichlet_distribution.cpp
    src/conjugate_categorical_dirichlet.cpp 
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

# ======== Tests: C++ ========

# For testing C++, we use FetchContent to get Google Test (gtest)
include(FetchContent)

# Fetch Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0  # Use latest stable release
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Create test executables
add_executable(test_tree tests/test_tree.cpp)
target_link_libraries(test_tree PRIVATE bayes_tree gtest_main)

add_executable(test_categorical_distribution tests/test_categorical_distribution.cpp)
target_link_libraries(test_categorical_distribution PRIVATE bayes_tree gtest_main)

add_executable(test_dirichlet_distribution tests/test_dirichlet_distribution.cpp)
target_link_libraries(test_dirichlet_distribution PRIVATE bayes_tree gtest_main)

add_executable(test_conjugate_categorical_dirichlet tests/test_conjugate_categorical_dirichlet.cpp)
target_link_libraries(test_conjugate_categorical_dirichlet PRIVATE bayes_tree gtest_main)

# Auto-discover tests using gtest_discover_tests
include(GoogleTest)
gtest_discover_tests(test_tree)
gtest_discover_tests(test_categorical_distribution)
gtest_discover_tests(test_dirichlet_distribution)
gtest_discover_tests(test_conjugate_categorical_dirichlet)





# ======== Python bindings ========
# Create CMake cache variable and set default value
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON) 
if (BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()



