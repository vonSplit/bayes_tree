// test_dirichlet.cpp
#include "bayes_tree/dirichlet_distribution.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <numeric>

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

void test_constructor() {
    std::cout << "\n=== Constructor Tests ===\n";
    
    // Valid construction
    try {
        std::vector<double> alpha = {1.0, 2.0, 3.0};
        DirichletDistribution d(alpha);
        test("Constructor with valid alpha", true);
    } catch (...) {
        test("Constructor with valid alpha", false);
    }
    
    // Empty alpha
    try {
        std::vector<double> alpha = {};
        DirichletDistribution d(alpha);
        test("Constructor rejects empty alpha", false);
    } catch (const std::invalid_argument&) {
        test("Constructor rejects empty alphaas expected", true);
    }
    
    // Negative alpha
    try {
        std::vector<double> alpha = {1.0, -2.0, 3.0};
        DirichletDistribution d(alpha);
        test("Constructor rejects negative alpha", false);
    } catch (const std::invalid_argument&) {
        test("Constructor rejects negative alpha", true);
    }
    
    // Zero alpha
    try {
        std::vector<double> alpha = {1.0, 0.0, 3.0};
        DirichletDistribution d(alpha);
        test("Constructor rejects zero alpha", false);
    } catch (const std::invalid_argument&) {
        test("Constructor rejects zero alpha", true);
    }
}

void test_dimension() {
    std::cout << "\n=== Dimension Tests ===\n";
    
    std::vector<double> alpha = {1.0, 2.0, 3.0, 4.0};
    DirichletDistribution d(alpha);
    test("Dimension returns correct size", d.dimension() == 4);
}

void test_mean() {
    std::cout << "\n=== Mean Tests ===\n";
    
    // Uniform distribution
    std::vector<double> alpha1 = {1.0, 1.0, 1.0};
    DirichletDistribution d1(alpha1);
    auto mean1 = d1.mean();
    std::vector<double> expected1 = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    test("Mean of uniform Dirichlet", vector_approx_equal(mean1, expected1));
    
    // Non-uniform distribution
    std::vector<double> alpha2 = {2.0, 3.0, 5.0};
    DirichletDistribution d2(alpha2);
    auto mean2 = d2.mean();
    std::vector<double> expected2 = {0.2, 0.3, 0.5};
    test("Mean of non-uniform Dirichlet", vector_approx_equal(mean2, expected2));
}

void test_variance() {
    std::cout << "\n=== Variance Tests ===\n";
    
    std::vector<double> alpha = {2.0, 3.0, 5.0};
    DirichletDistribution d(alpha);
    auto var = d.variance();
    
    // Check variance calculation: Var[X_i] = alpha_i(alpha_sum - alpha_i) / (alpha_sum^2 * (alpha_sum + 1))
    // JT: Would be nicer as p(1-p)/(alpha_sum+1) but ok for now
    double alpha_sum = 10.0;
    std::vector<double> expected_var = {
        (2.0 * (alpha_sum-2.0)) / (alpha_sum * alpha_sum * (alpha_sum + 1.0)),
        (3.0 * (alpha_sum-3.0)) / (alpha_sum * alpha_sum * (alpha_sum + 1.0)),
        (5.0 * (alpha_sum-5.0)) / (alpha_sum * alpha_sum * (alpha_sum + 1.0))
    };
    
    test("Variance calculation correct", vector_approx_equal(var, expected_var));
}

void test_sample() {
    std::cout << "\n=== Sample Tests ===\n";
    
    std::vector<double> alpha = {2.0, 3.0, 5.0};
    DirichletDistribution d(alpha);
    
    // Test single sample
    auto sample = d.sample();
    test("Sample has correct dimension", sample.size() == 3);
    
    double sum = std::accumulate(sample.begin(), sample.end(), 0.0);
    test("Sample sums to 1", approx_equal(sum, 1.0));
    
    bool all_positive = true;
    for (double val : sample) {
        if (val <= 0.0 || val >= 1.0) {
            all_positive = false;
            break;
        }
    }
    test("Sample values in (0, 1)", all_positive);
    
    // Test multiple samples
    auto samples = d.sample(10);
    test("Multiple samples returns correct count", samples.size() == 10);
    test("Each sample has correct dimension", samples[0].size() == 3);
}

void test_sample_statistics() {
    std::cout << "\n=== Sample Statistics Tests ===\n";
    
    std::vector<double> alpha = {5.0, 5.0, 5.0};
    DirichletDistribution d(alpha);
    
    // Generate many samples and check if empirical mean converges to theoretical mean
    const size_t n_samples = 10000;
    std::vector<double> sum(3, 0.0);
    
    for (size_t i = 0; i < n_samples; ++i) {
        auto sample = d.sample();
        for (size_t j = 0; j < 3; ++j) {
            sum[j] += sample[j];
        }
    }
    
    std::vector<double> empirical_mean(3);
    for (size_t i = 0; i < 3; ++i) {
        empirical_mean[i] = sum[i] / n_samples;
    }
    
    auto theoretical_mean = d.mean();
    test("Empirical mean converges to theoretical mean", 
         vector_approx_equal(empirical_mean, theoretical_mean, 0.01));
}

void test_log_pdf() {
    std::cout << "\n=== Log PDF Tests ===\n";
    
    std::vector<double> alpha = {2.0, 3.0, 5.0};
    DirichletDistribution d(alpha);
    
    // Valid point
    std::vector<double> x1 = {0.2, 0.3, 0.5};
    try {
        double logp = d.logPdf(x1);
        test("Log PDF computes for valid point", std::isfinite(logp));
    } catch (...) {
        test("Log PDF computes for valid point", false);
    }
    
    // Point that doesn't sum to 1
    std::vector<double> x2 = {0.2, 0.3, 0.4};
    try {
        d.logPdf(x2);
        test("Log PDF rejects point not summing to 1", false);
    } catch (const std::invalid_argument&) {
        test("Log PDF rejects point not summing to 1", true);
    }
    
    // Wrong dimension
    std::vector<double> x3 = {0.5, 0.5};
    try {
        d.logPdf(x3);
        test("Log PDF rejects wrong dimension", false);
    } catch (const std::invalid_argument&) {
        test("Log PDF rejects wrong dimension", true);
    }
    
    // Boundary case (should return -infinity)
    std::vector<double> x4 = {0.0, 0.5, 0.5};
    double logp = d.logPdf(x4);
    test("Log PDF returns -inf for boundary point", std::isinf(logp) && logp < 0);
    
    // Mean should have higher probability than extreme point
    auto mean_point = d.mean();
    std::vector<double> extreme_point = {0.01, 0.01, 0.98};
    test("Log PDF higher at mean than extreme point", 
         d.logPdf(mean_point) > d.logPdf(extreme_point));
}

void test_get_set_alpha() {
    std::cout << "\n=== Get/Set Alpha Tests ===\n";
    
    std::vector<double> alpha = {2.0, 3.0, 5.0};
    DirichletDistribution d(alpha);
    
    auto retrieved = d.getAlpha();
    test("Get alpha returns correct values", vector_approx_equal(retrieved, alpha));
    
    // Set valid alpha
    std::vector<double> new_alpha = {1.0, 2.0, 3.0};
    try {
        d.setAlpha(new_alpha);
        test("Set alpha with valid values", vector_approx_equal(d.getAlpha(), new_alpha));
    } catch (...) {
        test("Set alpha with valid values", false);
    }
    
    // Set alpha with wrong size
    try {
        std::vector<double> wrong_size = {1.0, 2.0};
        d.setAlpha(wrong_size);
        test("Set alpha rejects wrong size", false);
    } catch (const std::invalid_argument&) {
        test("Set alpha rejects wrong size", true);
    }
    
    // Set alpha with invalid values
    try {
        std::vector<double> invalid = {1.0, -2.0, 3.0};
        d.setAlpha(invalid);
        test("Set alpha rejects negative values", false);
    } catch (const std::invalid_argument&) {
        test("Set alpha rejects negative values", true);
    }
}

int main() {
    std::cout << "Running Dirichlet Distribution Unit Tests\n";
    std::cout << "==========================================\n";
    
    test_constructor();
    test_dimension();
    test_mean();
    test_variance();
    test_sample();
    test_sample_statistics();
    test_log_pdf();
    test_get_set_alpha();
    
    std::cout << "\n==========================================\n";
    std::cout << "Test Results: " << passed_tests << "/" << total_tests << " passed\n";
    
    if (passed_tests == total_tests) {
        std::cout << "All tests passed! ✓\n";
        return 0;
    } else {
        std::cout << (total_tests - passed_tests) << " test(s) failed! ✗\n";
        return 1;
    }
}