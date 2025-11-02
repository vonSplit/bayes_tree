#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for automatic conversion of std::vector <-> Python lists
#include "bayes_tree/bayes_tree.hpp"
#include "bayes_tree/dirichlet_distribution.hpp"
#include "bayes_tree/conjugate_categorical_dirichlet.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pybayes_tree, m) {
    py::class_<BayesTree>(m, "BayesTree")
        .def(py::init<>())
        .def("predict", &BayesTree::predict);

    py::class_<DirichletDistribution>(m, "DirichletDistribution")
        .def(py::init<const std::vector<double>&, unsigned int>(),
             py::arg("alpha"), py::arg("seed") = std::random_device{}())
        .def("sample"   , py::overload_cast<>      (&DirichletDistribution::sample, py::const_))
        .def("sample_n" , py::overload_cast<size_t>(&DirichletDistribution::sample, py::const_))
        .def("mean"     , &DirichletDistribution::mean     )
        .def("variance" , &DirichletDistribution::variance )
        .def("get_alpha", &DirichletDistribution::getAlpha, py::return_value_policy::reference_internal)
        .def("set_alpha", &DirichletDistribution::setAlpha )
        .def("dimension", &DirichletDistribution::dimension)
        .def("log_pdf"  , &DirichletDistribution::logPdf   );

         // ConjugateCategoricalDirichlet
     py::class_<ConjugateCategoricalDirichlet>(m, "ConjugateCategoricalDirichlet")
        .def(py::init<>())//;
        .def(py::init<int>())
        .def(py::init<int, double>())
        .def(py::init<const std::vector<double>&>())
        .def("initialise", [](ConjugateCategoricalDirichlet &self, int n) { self.initialise(n); })
        .def("initialise", [](ConjugateCategoricalDirichlet &self, int n, double alpha) { self.initialise(n, alpha); })
        .def("initialise", [](ConjugateCategoricalDirichlet &self, const std::vector<double>& v) { self.initialise(v); })
    //   .def("initialise", py::overload_cast<int>(&ConjugateCategoricalDirichlet::initialise))
    //   //  .def("initialise", py::overload_cast<int,double>(&ConjugateCategoricalDirichlet::initialise))
    //    // .def("initialise", py::overload_cast<const std::vector<double>&>(&ConjugateCategoricalDirichlet::initialise))
         .def("initialiseJeffreysFromObservationDistribution", &ConjugateCategoricalDirichlet::initialiseJeffreysFromObservationDistribution)
         .def("setJeffreysPrior", &ConjugateCategoricalDirichlet::setJeffreysPrior)
         .def("setAllParameterAlphasTo", &ConjugateCategoricalDirichlet::setAllParameterAlphasTo)
         .def("setJeffreysFromObservationDistribution", &ConjugateCategoricalDirichlet::setJeffreysFromObservationDistribution)
         .def("updateFromObservations", &ConjugateCategoricalDirichlet::updateFromObservations)
         .def("getLogLikelihoodFromObservations", &ConjugateCategoricalDirichlet::getLogLikelihoodFromObservations)
//     // Accessors: NEXT 2 LINES THROW ERRORS
      //   .def("getObservationDistribution", &ConjugateCategoricalDirichlet::getObservationDistribution, py::return_value_policy::reference)
       //    .def("getParameterDistribution", &ConjugateCategoricalDirichlet::getParameterDistribution, py::return_value_policy::reference)
           .def("getPriorType", &ConjugateCategoricalDirichlet::getPriorType)
           .def("getSingleAlpha", &ConjugateCategoricalDirichlet::getSingleAlpha)
           .def("getNumCategories", &ConjugateCategoricalDirichlet::getNumCategories)
           .def("getAlphas", &ConjugateCategoricalDirichlet::getAlphas)
        ;

    py::enum_<ConjugateCategoricalDirichlet::PriorType>(m, "PriorType")
        .value("Jeffreys", ConjugateCategoricalDirichlet::PriorType::Jeffreys)
        .value("EqualAlpha", ConjugateCategoricalDirichlet::PriorType::EqualAlpha)
        .value("ManualAlphas", ConjugateCategoricalDirichlet::PriorType::ManualAlphas)
        .value("ManualProbs", ConjugateCategoricalDirichlet::PriorType::ManualProbs);


}
