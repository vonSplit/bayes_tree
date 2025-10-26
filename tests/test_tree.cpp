#include "bayes_tree/bayes_tree.hpp"
#include <iostream>
#include <cassert>

int main() {
    BayesTree tree;
    double result = tree.predict(2.0);
    std::cout << "Prediction: " << result << std::endl;

    // simple test
    assert(result == 5.0);

    std::cout << "Test passed!" << std::endl;
    return 0;
}
