#pragma once

#include "dirichlet_distribution.hpp"
#include "categorical_distribution.hpp"
#include <vector>
#include <memory>

class ConjugateCategoricalDirichlet {
public:
    enum class PriorType {
        Jeffreys,      // 0.5 alpha for each category
        EqualAlpha,    // Same alpha for all categories
        ManualAlphas,  // Manually specified alphas
        ManualProbs    // Derived from categorical distribution
    };

    // Constructors
    ConjugateCategoricalDirichlet();
    explicit ConjugateCategoricalDirichlet(int num_categories);
    ConjugateCategoricalDirichlet(int num_categories, double alpha);
    explicit ConjugateCategoricalDirichlet(const std::vector<double>& alphas);
    
    // Copy constructor - initialise from priors only
    ConjugateCategoricalDirichlet(const ConjugateCategoricalDirichlet& other);
    
    // Assignment operator
    ConjugateCategoricalDirichlet& operator=(const ConjugateCategoricalDirichlet& other);
    
    // Initialise methods
    void initialise(int num_categories);
    void initialise(int num_categories, double alpha);
    void initialise(const std::vector<double>& alphas);
    void initialiseJeffreysFromObservationDistribution(const CategoricalDistribution& obs_dist);
    
    // Prior settings
    void setJeffreysPrior();
    void setAllParameterAlphasTo(double new_alpha);
    void setJeffreysFromObservationDistribution(const CategoricalDistribution& obs_dist);
    
    // Update from observations
    void updateFromObservations(const std::vector<int>& counts);
    
    // Log likelihood
    double getLogLikelihoodFromObservations(const std::vector<int>& counts) const;
    
    // Accessors
    const CategoricalDistribution& getObservationDistribution() const;
    const DirichletDistribution& getParameterDistribution() const;
    PriorType getPriorType() const;
    double getSingleAlpha() const;
    int getNumCategories() const;
    std::vector<double> getAlphas() const;
    
    // Mutators
    DirichletDistribution& getParameterDistribution();
    CategoricalDistribution& getObservationDistribution();

private:
    void updateObservationDistribution();
    double gammaLn(double x) const;
    
    PriorType prior_type_;
    double single_alpha_;  // For EqualAlpha or Jeffreys priors
    std::vector<double> manual_alphas_;  // For ManualAlphas prior
    
    std::unique_ptr<DirichletDistribution> parameter_distribution_;
    std::unique_ptr<CategoricalDistribution> observation_distribution_;
};