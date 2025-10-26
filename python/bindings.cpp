#include <pybind11/pybind11.h>
#include "bayes_tree/bayes_tree.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pybayes_tree, m) {
    py::class_<BayesTree>(m, "BayesTree")
        .def(py::init<>())
        .def("predict", &BayesTree::predict);
}
