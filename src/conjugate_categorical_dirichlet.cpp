#include "bayes_tree/conjugate_categorical_dirichlet.hpp"
#include <stdexcept>
#include <cmath>
#include <numeric>

// Default constructor - Jeffreys prior with 2 categories
ConjugateCategoricalDirichlet::ConjugateCategoricalDirichlet() {
    initialise(2);
}

// Constructor with number of categories (Jeffreys prior)
ConjugateCategoricalDirichlet::ConjugateCategoricalDirichlet(int num_categories) {
    initialise(num_categories);
}

// Constructor with equal alpha for all categories
ConjugateCategoricalDirichlet::ConjugateCategoricalDirichlet(int num_categories, double alpha) {
    initialise(num_categories, alpha);
}

// Constructor with manual alphas
ConjugateCategoricalDirichlet::ConjugateCategoricalDirichlet(const std::vector<double>& alphas) {
    initialise(alphas);
}

// Copy constructor - initialise from priors only
ConjugateCategoricalDirichlet::ConjugateCategoricalDirichlet(const ConjugateCategoricalDirichlet& other)
    : prior_type_(other.prior_type_)
    , single_alpha_(other.single_alpha_)
    , manual_alphas_(other.manual_alphas_)
{
    parameter_distribution_ = std::make_unique<DirichletDistribution>(*other.parameter_distribution_);
    observation_distribution_ = std::make_unique<CategoricalDistribution>(*other.observation_distribution_);
}

// Assignment operator
ConjugateCategoricalDirichlet& ConjugateCategoricalDirichlet::operator=(const ConjugateCategoricalDirichlet& other) {
    if (this != &other) {
        prior_type_ = other.prior_type_;
        single_alpha_ = other.single_alpha_;
        manual_alphas_ = other.manual_alphas_;
        parameter_distribution_ = std::make_unique<DirichletDistribution>(*other.parameter_distribution_);
        observation_distribution_ = std::make_unique<CategoricalDistribution>(*other.observation_distribution_);
    }
    return *this;
}

// Initialise with Jeffreys prior (0.5 alpha for each category)
void ConjugateCategoricalDirichlet::initialise(int num_categories) {
    prior_type_ = PriorType::Jeffreys;
    single_alpha_ = 0.5;
    
    std::vector<double> alphas(num_categories, single_alpha_);
    parameter_distribution_ = std::make_unique<DirichletDistribution>(alphas);
    
    auto means = parameter_distribution_->mean();
    observation_distribution_ = std::make_unique<CategoricalDistribution>(means);
}

// Initialise with equal alpha for all categories
void ConjugateCategoricalDirichlet::initialise(int num_categories, double alpha) {
    std::vector<double> alphas(num_categories, alpha);
    initialise(alphas);
    
    prior_type_ = PriorType::EqualAlpha;
    single_alpha_ = alpha;
}

// Initialise with manual alphas
void ConjugateCategoricalDirichlet::initialise(const std::vector<double>& alphas) {
    parameter_distribution_ = std::make_unique<DirichletDistribution>(alphas);
    
    auto means = parameter_distribution_->mean();
    observation_distribution_ = std::make_unique<CategoricalDistribution>(means);
    
    prior_type_ = PriorType::ManualAlphas;
    manual_alphas_ = alphas;
}

// Initialise Jeffreys prior from observation distribution
void ConjugateCategoricalDirichlet::initialiseJeffreysFromObservationDistribution(
    const CategoricalDistribution& obs_dist) {
    
    prior_type_ = PriorType::ManualProbs;
    single_alpha_ = -1.0;
    
    int num_categories = obs_dist.probs().size();
    
    // Set total prior alpha to 0.5 per category (mirroring Jeffreys prior)
    double total_alpha = num_categories * 0.5;
    
    // Scale the observation probabilities by total_alpha to get alphas
    std::vector<double> alphas(num_categories);
    const auto& probs = obs_dist.probs();
    for (int i = 0; i < num_categories; ++i) {
        alphas[i] = probs[i] * total_alpha;
    }
    
    parameter_distribution_ = std::make_unique<DirichletDistribution>(alphas);
    
    auto means = parameter_distribution_->mean();
    observation_distribution_ = std::make_unique<CategoricalDistribution>(means);
}

void ConjugateCategoricalDirichlet::setJeffreysPrior() {
    initialise(parameter_distribution_->dimension());
}

void ConjugateCategoricalDirichlet::setAllParameterAlphasTo(double new_alpha) {
    initialise(parameter_distribution_->dimension(), new_alpha);
}

void ConjugateCategoricalDirichlet::setJeffreysFromObservationDistribution(
    const CategoricalDistribution& obs_dist) {
    initialiseJeffreysFromObservationDistribution(obs_dist);
}

// Update from observed counts
void ConjugateCategoricalDirichlet::updateFromObservations(const std::vector<int>& counts) {
    int num_categories = parameter_distribution_->dimension();
    
    if (num_categories != static_cast<int>(counts.size())) {
        throw std::invalid_argument(
            "Length of observed value vector doesn't match distribution dimension");
    }
    
    // Get current alphas and add counts
    auto current_alphas = parameter_distribution_->getAlpha();
    std::vector<double> new_alphas(num_categories);
    
    for (int i = 0; i < num_categories; ++i) {
        new_alphas[i] = current_alphas[i] + counts[i];
    }
    
    // Update parameter distribution
    parameter_distribution_->setAlpha(new_alphas);
    
    // Update observation distribution with new means
    updateObservationDistribution();
}

// Calculate log likelihood from observed counts
double ConjugateCategoricalDirichlet::getLogLikelihoodFromObservations(
    const std::vector<int>& counts) const {
    
    int num_categories = parameter_distribution_->dimension();
    
    if (num_categories != static_cast<int>(counts.size())) {
        throw std::invalid_argument(
            "Length of observed value vector doesn't match distribution dimension");
    }
    
    auto alphas = parameter_distribution_->getAlpha();
    
    double alpha_total = 0.0;
    double count_total = 0.0;
    double log_likelihood = 0.0;
    
    for (int i = 0; i < num_categories; ++i) {
        count_total += counts[i];
        alpha_total += alphas[i];
        
        log_likelihood += gammaLn(counts[i] + alphas[i]) - gammaLn(alphas[i]);
    }
    
    log_likelihood += gammaLn(alpha_total) - gammaLn(count_total + alpha_total);
    
    return log_likelihood;
}

// Accessors
const CategoricalDistribution& ConjugateCategoricalDirichlet::getObservationDistribution() const {
    return *observation_distribution_;
}

const DirichletDistribution& ConjugateCategoricalDirichlet::getParameterDistribution() const {
    return *parameter_distribution_;
}

ConjugateCategoricalDirichlet::PriorType ConjugateCategoricalDirichlet::getPriorType() const {
    return prior_type_;
}

double ConjugateCategoricalDirichlet::getSingleAlpha() const {
    if (prior_type_ == PriorType::EqualAlpha || prior_type_ == PriorType::Jeffreys) {
        return single_alpha_;
    }
    throw std::logic_error("No single alpha defined for ManualAlphas or ManualProbs prior type");
}

int ConjugateCategoricalDirichlet::getNumCategories() const {
    return observation_distribution_->probs().size();
}

std::vector<double> ConjugateCategoricalDirichlet::getAlphas() const {
    return parameter_distribution_->getAlpha();
}

// Mutators
DirichletDistribution& ConjugateCategoricalDirichlet::getParameterDistribution() {
    return *parameter_distribution_;
}

CategoricalDistribution& ConjugateCategoricalDirichlet::getObservationDistribution() {
    return *observation_distribution_;
}

// Private methods
void ConjugateCategoricalDirichlet::updateObservationDistribution() {
    auto means = parameter_distribution_->mean();
    observation_distribution_ = std::make_unique<CategoricalDistribution>(means);
}

// Log gamma function (using standard library)
double ConjugateCategoricalDirichlet::gammaLn(double x) const {
    return std::lgamma(x);
}