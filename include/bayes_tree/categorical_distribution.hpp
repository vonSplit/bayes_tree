#pragma once
#include <vector>

class CategoricalDistribution {
public:
    CategoricalDistribution(const std::vector<double>& probs);  // declaration only
    const std::vector<double>& probs() const;
    double log_likelihood(const std::vector<int>& counts) const;

private:
    void normalize();
    std::vector<double> probabilities_;
};