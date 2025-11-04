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

# For testing C++, we use FetchContent to get Catch2
include(FetchContent)

# Fetch Catch2 (header-only, so no external build step)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0  # or latest stable
)
FetchContent_MakeAvailable(Catch2)

enable_testing()
include(Catch)  # provides catch_discover_tests()
# Create test executable test_tree from the test source file
# test_tree has access to bayes_tree library
#target_link_libraries(test_tree PRIVATE bayes_tree)

add_executable       (test_tree tests/test_tree.cpp)
target_link_libraries(test_tree PRIVATE bayes_tree Catch2::Catch2WithMain)

add_executable       (test_categorical_distribution tests/test_categorical_distribution.cpp)
target_link_libraries(test_categorical_distribution PRIVATE bayes_tree Catch2::Catch2WithMain)
# target_link_libraries(test_categorical_distribution PRIVATE bayes_tree)

# add_executable       (test_dirichlet_distribution tests/test_dirichlet_distribution.cpp)
# target_link_libraries(test_dirichlet_distribution PRIVATE bayes_tree)

# add_executable       (test_conjugate_categorical_dirichlet tests/test_dirichlet_distribution.cpp)
# target_link_libraries(test_conjugate_categorical_dirichlet PRIVATE bayes_tree)

# Enable CTest support  
enable_testing()
# Register tests with CTest: NAME <test_name> COMMAND <executable>
# Or can discover autoatically with Catch2: catch_discover_tests()
#add_test(NAME test_tree                            COMMAND test_tree)
#add_test(NAME test_categorical_distribution        COMMAND test_categorical_distribution)
#add_test(NAME test_dirichlet_distribution          COMMAND test_dirichlet_distribution)
#add_test(NAME test_conjugate_categorical_dirichlet COMMAND test_conjugate_categorical_dirichlet)

#add_executable(test_core tests/test_core.cpp)
#target_link_libraries(test_core PRIVATE Catch2::Catch2WithMain)

# This line auto-discovers all TEST_CASEs in the binary and adds them to CTest
catch_discover_tests(test_tree)
catch_discover_tests(test_categorical_distribution) 
#catch_discover_tests(test_categorical_distribution)
#catch_discover_tests(test_dirichlet_distribution)
#catch_discover_tests(test_conjugate_categorical_dirichlet)





# ======== Python bindings ========
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if (BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()



