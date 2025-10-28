#include "bayes_tree/categorical_distribution.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <iostream>

CategoricalDistribution::CategoricalDistribution(const std::vector<double>& probs) {
    std::cout << "Constructor called\n";
    if (probs.empty())
        throw std::invalid_argument("Probability vector cannot be empty.");

    probabilities_ = probs;  
    normalize();
}

// Normalizes the probability vector to sum to 1
void CategoricalDistribution::normalize() {
    double sum = std::accumulate(probabilities_.begin(), probabilities_.end(), 0.0);
    if (sum <= 0.0)
        throw std::invalid_argument("Sum of probabilities must be positive.");

    for (auto& p : probabilities_)
        p /= sum;
}

// Return probabilities
const std::vector<double>& CategoricalDistribution::probs() const {
    return probabilities_;
}

// Compute log-likelihood given counts for each category
double CategoricalDistribution::log_likelihood(const std::vector<int>& counts) const {
    if (counts.size() != probabilities_.size())
        throw std::invalid_argument("Counts and probability vectors must be same length.");

    double loglike = 0.0;
    for (size_t i = 0; i < counts.size(); ++i) {
        if (probabilities_[i] <= 0.0 && counts[i] > 0)
            return -INFINITY;  // impossible outcome
        if (counts[i] > 0)
            loglike += counts[i] * std::log(probabilities_[i]);
    }
    return loglike;
}

//std::vector<double> probabilities_;
