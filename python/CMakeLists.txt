# Python bindings subproject
set(pybind11_DIR "C:/Users/User/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages/pybind11/share/cmake/pybind11")
cmake_policy(SET CMP0148 OLD)
find_package(pybind11 CONFIG REQUIRED)

# Create Python extension module
pybind11_add_module(pybayes_tree bindings.cpp)
target_link_libraries(pybayes_tree PRIVATE bayes_tree)

## Optional: control output folder
#set_target_properties(pybayes_tree PROPERTIES
#    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
#)
