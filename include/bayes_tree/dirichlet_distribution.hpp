// DirichletDistribution.hpp
#pragma once

#include <vector>
#include <random>

class DirichletDistribution {
private:
    std::vector<double> alpha;
    mutable std::mt19937 gen;
    
public:
    // Constructor with concentration parameters
    DirichletDistribution(const std::vector<double>& concentration_params, 
                            unsigned int seed = std::random_device{}());
    
    // Generate a sample from the Dirichlet distribution
    std::vector<double> sample() const;
    
    // Generate multiple samples
    std::vector<std::vector<double>> sample(size_t n) const;
    
    // Get mean of the distribution
    std::vector<double> mean() const;
    
    // Get variance for each component
    std::vector<double> variance() const;
    
    // Get concentration parameters
    const std::vector<double>& getAlpha() const;
    
    // Set new concentration parameters
    void setAlpha(const std::vector<double>& new_alpha);
    
    // Get dimensionality
    size_t dimension() const;
    
    // Compute log probability density (up to normalization constant)
    double logPdf(const std::vector<double>& x) const;
};
