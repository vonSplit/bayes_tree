find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pybayes_tree bindings.cpp)
target_link_libraries(pybayes_tree PRIVATE bayes_tree)
