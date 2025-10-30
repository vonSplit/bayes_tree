// DirichletDistribution.cpp
#include "bayes_tree/dirichlet_distribution.hpp"
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <limits>

DirichletDistribution::DirichletDistribution(
    const std::vector<double>& concentration_params, 
    unsigned int seed)
    : alpha(concentration_params), gen(seed) {
    if (alpha.empty()) {
        throw std::invalid_argument("Concentration parameters cannot be empty");
    }
    for (double a : alpha) {
        if (a <= 0.0) {
            throw std::invalid_argument("All concentration parameters must be positive");
        }
    }
}

std::vector<double> DirichletDistribution::sample() const {
    std::vector<double> result(alpha.size());
    double sum = 0.0;
    
    // Sample from gamma distributions and normalize
    for (size_t i = 0; i < alpha.size(); ++i) {
        std::gamma_distribution<double> gamma_dist(alpha[i], 1.0);
        result[i] = gamma_dist(gen);
        sum += result[i];
    }
    
    // Normalize to sum to 1
    for (double& val : result) {
        val /= sum;
    }
    
    return result;
}

std::vector<std::vector<double>> DirichletDistribution::sample(size_t n) const {
    std::vector<std::vector<double>> samples;
    samples.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        samples.push_back(sample());
    }
    return samples;
}

std::vector<double> DirichletDistribution::mean() const {
    double alpha_sum = std::accumulate(alpha.begin(), alpha.end(), 0.0);
    std::vector<double> m(alpha.size());
    for (size_t i = 0; i < alpha.size(); ++i) {
        m[i] = alpha[i] / alpha_sum;
    }
    return m;
}

std::vector<double> DirichletDistribution::variance() const {
    double alpha_sum = std::accumulate(alpha.begin(), alpha.end(), 0.0);
    std::vector<double> var(alpha.size());
    for (size_t i = 0; i < alpha.size(); ++i) {
        var[i] = (alpha[i] * (alpha_sum - alpha[i])) / 
                 (alpha_sum * alpha_sum * (alpha_sum + 1.0));
    }
    return var;
}

const std::vector<double>& DirichletDistribution::getAlpha() const {
    return alpha;
}

void DirichletDistribution::setAlpha(const std::vector<double>& new_alpha) {
    if (new_alpha.size() != alpha.size()) {
        throw std::invalid_argument("New alpha must have same size as original");
    }
    for (double a : new_alpha) {
        if (a <= 0.0) {
            throw std::invalid_argument("All concentration parameters must be positive");
        }
    }
    alpha = new_alpha;
}

size_t DirichletDistribution::dimension() const {
    return alpha.size();
}

double DirichletDistribution::logPdf(const std::vector<double>& x) const {
    if (x.size() != alpha.size()) {
        throw std::invalid_argument("Input dimension mismatch");
    }
    
    double sum = std::accumulate(x.begin(), x.end(), 0.0);
    if (std::abs(sum - 1.0) > 1e-6) {
        throw std::invalid_argument("Input must sum to 1");
    }
    
    double log_prob = 0.0;
    for (size_t i = 0; i < alpha.size(); ++i) {
        if (x[i] <= 0.0 || x[i] >= 1.0) {
            return -std::numeric_limits<double>::infinity();
        }
        log_prob += (alpha[i] - 1.0) * std::log(x[i]);
    }
    
    return log_prob;
}