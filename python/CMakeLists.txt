cmake_minimum_required(VERSION 3.20)
project(bayes_tree_python_bindings LANGUAGES CXX)

# -----------------------------
# 1. Detect active Python
# -----------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Optional: explicitly tell pybind11 which Python executable to use
set(PYBIND11_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

# -----------------------------
# 2. Find pybind11
# -----------------------------
# If using conda, automatically append pybind11 path
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "JT: conda prefix is defined")
    message(STATUS "JT: conda prefix = $ENV{CONDA_PREFIX}")
else()
    message(STATUS "JT: conda prefix is not defined")
endif()
if(DEFINED pybind11_DIR)
    message(STATUS "JT: pybind11_DIR is defined")
    message(STATUS "JT: pybind11_DIR = ${pybind11_DIR}")
else()
    message(STATUS "JT: pybind11_DIR is not defined")
endif()
if((NOT DEFINED pybind11_DIR OR pybind11_DIR STREQUAL "pybind11_DIR-NOTFOUND") AND DEFINED ENV{CONDA_PREFIX})
    message(STATUS "JT: Setting pybind11_DIR based on CONDA_PREFIX")
#    set(pybind11_DIR "$ENV{CONDA_PREFIX}/Lib/site-packages/pybind11/share/cmake/pybind11")
    # Use backslashes for Windows-style paths, and escape them properly
    set(pybind11_DIR "$ENV{CONDA_PREFIX}\\Lib\\site-packages\\pybind11\\share\\cmake\\pybind11")
    message(STATUS "JT: pybind11_DIR = ${pybind11_DIR}")
endif()
find_package(pybind11 CONFIG REQUIRED)

# -----------------------------
# 3. Add the Python module
# -----------------------------
pybind11_add_module(pybayes_tree bindings.cpp)
target_link_libraries(pybayes_tree PRIVATE bayes_tree)

# -----------------------------
# 4. Force output to build/python
# Works for multi-config (MSVC) and single-config (Ninja)
# -----------------------------
set_target_properties(pybayes_tree PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/python"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/python"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/python"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/python"
)
