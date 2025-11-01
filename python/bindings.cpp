#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for automatic conversion of std::vector <-> Python lists
#include "bayes_tree/bayes_tree.hpp"
#include "bayes_tree/dirichlet_distribution.hpp"

//pybayes_tree.cp311-win_amd64.pyd

namespace py = pybind11;

PYBIND11_MODULE(pybayes_tree, m) {
    py::class_<BayesTree>(m, "BayesTree")
        .def(py::init<>())
        .def("predict", &BayesTree::predict);

    py::class_<DirichletDistribution>(m, "DirichletDistribution")
        .def(py::init<const std::vector<double>&, unsigned int>(),
             py::arg("alpha"), py::arg("seed") = std::random_device{}())
        .def("sample", py::overload_cast<>(&DirichletDistribution::sample, py::const_))
        .def("sample_n", py::overload_cast<size_t>(&DirichletDistribution::sample, py::const_))
        .def("mean", &DirichletDistribution::mean)
        .def("variance", &DirichletDistribution::variance)
        .def("get_alpha", &DirichletDistribution::getAlpha, py::return_value_policy::reference_internal)
        .def("set_alpha", &DirichletDistribution::setAlpha)
        .def("dimension", &DirichletDistribution::dimension)
        .def("log_pdf", &DirichletDistribution::logPdf);
}
