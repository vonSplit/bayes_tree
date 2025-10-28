#include <iostream>
#include <cassert>
#include "bayes_tree/bayes_tree.hpp"

int main() {
    BayesTree tree;
    double result = tree.predict(2.0);
    std::cout << "Prediction: " << result << std::endl;
    //assert(false);
    // simple test
    assert(result == 5.0);

    std::cout << "Test passed!" << std::endl;
    return 0;
}
