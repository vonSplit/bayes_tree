#include "bayes_tree/bayes_tree.hpp"
#include <iostream>

// Define the constructor
BayesTree::BayesTree() {
    std::cout << "BayesTree constructed.\n";
}

// Define the predict function
double BayesTree::predict(double x) const {
    return 2.0 * x + 1.0;
}
