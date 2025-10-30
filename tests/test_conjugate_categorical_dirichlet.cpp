#include "bayes_tree/conjugate_categorical_dirichlet.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

// Simple test framework
int total_tests = 0;
int passed_tests = 0;

void test(const std::string& name, bool condition) {
    total_tests++;
    if (condition) {
        passed_tests++;
        std::cout << "[PASS] " << name << "\n";
    } else {
        std::cout << "[FAIL] " << name << "\n";
    }
}

bool approx_equal(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) < eps;
}

bool vector_approx_equal(const std::vector<double>& a, const std::vector<double>& b, double eps = 1e-6) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!approx_equal(a[i], b[i], eps)) return false;
    }
    return true;
}

void test_constructors() {
    std::cout << "\n=== Constructor Tests ===\n";
    
    // Default constructor
    try {
        ConjugateCategoricalDirichlet cd1;
        test("Default constructor (2 categories)", cd1.getNumCategories() == 2);
        test("Default constructor uses Jeffreys prior", 
             cd1.getPriorType() == ConjugateCategoricalDirichlet::PriorType::Jeffreys);
    } catch (...) {
        test("Default constructor", false);
    }
    
    // Constructor with num categories
    try {
        ConjugateCategoricalDirichlet cd2(5);
        test("Constructor with num categories", cd2.getNumCategories() == 5);
        test("Jeffreys prior has alpha=0.5", approx_equal(cd2.getSingleAlpha(), 0.5));
    } catch (...) {
        test("Constructor with num categories", false);
    }
    
    // Constructor with equal alpha
    try {
        ConjugateCategoricalDirichlet cd3(3, 2.0);
        test("Constructor with equal alpha", cd3.getNumCategories() == 3);
        test("Equal alpha stored correctly", approx_equal(cd3.getSingleAlpha(), 2.0));
        test("Prior type is EqualAlpha", 
             cd3.getPriorType() == ConjugateCategoricalDirichlet::PriorType::EqualAlpha);
    } catch (...) {
        test("Constructor with equal alpha", false);
    }
    
    // Constructor with manual alphas
    try {
        std::vector<double> alphas = {1.0, 2.0, 3.0, 4.0};
        ConjugateCategoricalDirichlet cd4(alphas);
        test("Constructor with manual alphas", cd4.getNumCategories() == 4);
        test("Prior type is ManualAlphas", 
             cd4.getPriorType() == ConjugateCategoricalDirichlet::PriorType::ManualAlphas);
        auto retrieved_alphas = cd4.getAlphas();
        test("Manual alphas stored correctly", vector_approx_equal(retrieved_alphas, alphas));
    } catch (...) {
        test("Constructor with manual alphas", false);
    }
}

void test_copy_constructor() {
    std::cout << "\n=== Copy Constructor Tests ===\n";
    
    std::vector<double> alphas = {2.0, 3.0, 5.0};
    ConjugateCategoricalDirichlet cd1(alphas);
    
    ConjugateCategoricalDirichlet cd2(cd1);
    
    test("Copy constructor preserves num categories", 
         cd2.getNumCategories() == cd1.getNumCategories());
    test("Copy constructor preserves prior type", 
         cd2.getPriorType() == cd1.getPriorType());
    test("Copy constructor preserves alphas", 
         vector_approx_equal(cd2.getAlphas(), cd1.getAlphas()));
}

void test_initialization() {
    std::cout << "\n=== Initialization Tests ===\n";
    
    ConjugateCategoricalDirichlet cd(3);
    
    // Test Jeffreys initialization
    cd.setJeffreysPrior();
    test("Set Jeffreys prior", 
         cd.getPriorType() == ConjugateCategoricalDirichlet::PriorType::Jeffreys);
    test("Jeffreys prior alpha is 0.5", approx_equal(cd.getSingleAlpha(), 0.5));
    
    // Test setting all alphas
    cd.setAllParameterAlphasTo(3.0);
    test("Set all alphas", approx_equal(cd.getSingleAlpha(), 3.0));
    test("Prior type changed to EqualAlpha", 
         cd.getPriorType() == ConjugateCategoricalDirichlet::PriorType::EqualAlpha);
}

void test_observation_distribution() {
    std::cout << "\n=== Observation Distribution Tests ===\n";
    
    // Uniform prior should give uniform observation distribution
    ConjugateCategoricalDirichlet cd(3);
    const auto& obs_dist = cd.getObservationDistribution();
    const auto& probs = obs_dist.probs();
    
    test("Observation distribution has correct size", probs.size() == 3);
    
    std::vector<double> expected = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    test("Jeffreys prior gives uniform observation distribution", 
         vector_approx_equal(probs, expected));
    
    // Non-uniform prior
    std::vector<double> alphas = {2.0, 3.0, 5.0};
    ConjugateCategoricalDirichlet cd2(alphas);
    const auto& obs_dist2 = cd2.getObservationDistribution();
    const auto& probs2 = obs_dist2.probs();
    
    std::vector<double> expected2 = {0.2, 0.3, 0.5};
    test("Non-uniform alphas give correct observation distribution", 
         vector_approx_equal(probs2, expected2));
}

void test_update_from_observations() {
    std::cout << "\n=== Update from Observations Tests ===\n";
    
    std::vector<double> alphas = {2.0, 3.0, 5.0};
    ConjugateCategoricalDirichlet cd(alphas);
    
    // Get initial alphas
    auto initial_alphas = cd.getAlphas();
    
    // Update with counts
    std::vector<int> counts = {10, 20, 30};
    cd.updateFromObservations(counts);
    
    // Check updated alphas
    auto updated_alphas = cd.getAlphas();
    std::vector<double> expected = {12.0, 23.0, 35.0};
    
    test("Update adds counts to alphas", vector_approx_equal(updated_alphas, expected));
    
    // Check observation distribution updated
    const auto& obs_dist = cd.getObservationDistribution();
    const auto& probs = obs_dist.probs();
    
    double sum = 12.0 + 23.0 + 35.0;
    std::vector<double> expected_probs = {12.0/sum, 23.0/sum, 35.0/sum};
    
    test("Observation distribution updated after update", 
         vector_approx_equal(probs, expected_probs));
    
    // Test error on wrong size
    try {
        std::vector<int> wrong_size = {1, 2};
        cd.updateFromObservations(wrong_size);
        test("Update rejects wrong size counts", false);
    } catch (const std::invalid_argument&) {
        test("Update rejects wrong size counts", true);
    }
}

void test_log_likelihood() {
    std::cout << "\n=== Log Likelihood Tests ===\n";
    
    std::vector<double> alphas = {2.0, 3.0, 5.0};
    ConjugateCategoricalDirichlet cd(alphas);
    
    // Test log likelihood calculation
    std::vector<int> counts = {10, 15, 25};
    try {
        double log_lik = cd.getLogLikelihoodFromObservations(counts);
        test("Log likelihood computes without error", std::isfinite(log_lik));
    } catch (...) {
        test("Log likelihood computes without error", false);
    }
    
    // Test error on wrong size
    try {
        std::vector<int> wrong_size = {1, 2};
        cd.getLogLikelihoodFromObservations(wrong_size);
        test("Log likelihood rejects wrong size", false);
    } catch (const std::invalid_argument&) {
        test("Log likelihood rejects wrong size", true);
    }
    
    // More data should give higher likelihood for matching distribution
    std::vector<int> counts_small = {2, 3, 5};
    std::vector<int> counts_large = {20, 30, 50};
    
    double log_lik_small = cd.getLogLikelihoodFromObservations(counts_small);
    double log_lik_large = cd.getLogLikelihoodFromObservations(counts_large);
    
    test("Larger consistent sample has higher log likelihood", 
         log_lik_large > log_lik_small);
}

void test_jeffreys_from_observation_distribution() {
    std::cout << "\n=== Jeffreys from Observation Distribution Tests ===\n";
    
    std::vector<double> probs = {0.2, 0.3, 0.5};
    CategoricalDistribution cat_dist(probs);
    
    ConjugateCategoricalDirichlet cd(3);
    cd.setJeffreysFromObservationDistribution(cat_dist);
    
    test("Prior type set to ManualProbs", 
         cd.getPriorType() == ConjugateCategoricalDirichlet::PriorType::ManualProbs);
    
    // Check that observation distribution matches input
    const auto& obs_probs = cd.getObservationDistribution().probs();
    test("Observation distribution matches input distribution", 
         vector_approx_equal(obs_probs, probs, 0.01));
}

void test_sequential_updates() {
    std::cout << "\n=== Sequential Updates Tests ===\n";
    
    std::vector<double> alphas = {1.0, 1.0, 1.0};
    ConjugateCategoricalDirichlet cd(alphas);
    
    // Update in batches
    std::vector<int> batch1 = {5, 10, 15};
    std::vector<int> batch2 = {3, 7, 10};
    
    cd.updateFromObservations(batch1);
    cd.updateFromObservations(batch2);
    
    auto final_alphas = cd.getAlphas();
    std::vector<double> expected = {9.0, 18.0, 26.0};
    
    test("Sequential updates accumulate correctly", 
         vector_approx_equal(final_alphas, expected));
}

void test_accessors() {
    std::cout << "\n=== Accessor Tests ===\n";
    
    std::vector<double> alphas = {2.0, 3.0, 5.0};
    ConjugateCategoricalDirichlet cd(alphas);
    
    // Test const accessors
    const auto& const_cd = cd;
    
    try {
        const auto& obs_dist = const_cd.getObservationDistribution();
        const auto& par_dist = const_cd.getParameterDistribution();
        test("Const accessors work", true);
    } catch (...) {
        test("Const accessors work", false);
    }
    
    // Test non-const accessors
    try {
        auto& obs_dist = cd.getObservationDistribution();
        auto& par_dist = cd.getParameterDistribution();
        test("Non-const accessors work", true);
    } catch (...) {
        test("Non-const accessors work", false);
    }
}

int main() {
    std::cout << "Running Conjugate Categorical-Dirichlet Distribution Unit Tests\n";
    std::cout << "===============================================================\n";
    
    test_constructors();
    test_copy_constructor();
    test_initialization();
    test_observation_distribution();
    test_update_from_observations();
    test_log_likelihood();
    test_jeffreys_from_observation_distribution();
    test_sequential_updates();
    test_accessors();
    
    std::cout << "\n===============================================================\n";
    std::cout << "Test Results: " << passed_tests << "/" << total_tests << " passed\n";
    
    if (passed_tests == total_tests) {
        std::cout << "All tests passed! ✓\n";
        return 0;
    } else {
        std::cout << (total_tests - passed_tests) << " test(s) failed! ✗\n";
        return 1;
    }
}