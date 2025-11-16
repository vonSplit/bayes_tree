#include <gtest/gtest.h>
#include "bayes_tree/conjugate_categorical_dirichlet.hpp"
#include <cmath>
#include <numeric>

// Helper functions
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

// Test suite for constructors
class ConjugateCategoricalDirichletConstructorTest : public ::testing::Test {};

TEST_F(ConjugateCategoricalDirichletConstructorTest, DefaultConstructor) {
    ConjugateCategoricalDirichlet cd;
    EXPECT_EQ(cd.getNumCategories(), 2);
    EXPECT_EQ(cd.getPriorType(), ConjugateCategoricalDirichlet::PriorType::Jeffreys);
}

TEST_F(ConjugateCategoricalDirichletConstructorTest, ConstructorWithNumCategories) {
    ConjugateCategoricalDirichlet cd(5);
    EXPECT_EQ(cd.getNumCategories(), 5);
    EXPECT_TRUE(approx_equal(cd.getSingleAlpha(), 0.5));  // Jeffreys prior
}

TEST_F(ConjugateCategoricalDirichletConstructorTest, ConstructorWithEqualAlpha) {
    ConjugateCategoricalDirichlet cd(3, 2.0);
    EXPECT_EQ(cd.getNumCategories(), 3);
    EXPECT_TRUE(approx_equal(cd.getSingleAlpha(), 2.0));
    EXPECT_EQ(cd.getPriorType(), ConjugateCategoricalDirichlet::PriorType::EqualAlpha);
}

TEST_F(ConjugateCategoricalDirichletConstructorTest, ConstructorWithManualAlphas) {
    std::vector<double> alphas = {1.0, 2.0, 3.0, 4.0};
    ConjugateCategoricalDirichlet cd(alphas);
    EXPECT_EQ(cd.getNumCategories(), 4);
    EXPECT_EQ(cd.getPriorType(), ConjugateCategoricalDirichlet::PriorType::ManualAlphas);
    auto retrieved = cd.getAlphas();
    EXPECT_TRUE(vector_approx_equal(retrieved, alphas));
}

// Test suite for posterior update
class ConjugateCategoricalDirichletUpdateTest : public ::testing::Test {
protected:
    ConjugateCategoricalDirichlet cd{std::vector<double>{1.0, 1.0, 1.0}};
};

// TEST_F(ConjugateCategoricalDirichletUpdateTest, UpdateWithCounts) {
//     std::vector<int> counts = {5, 3, 2};
//     cd.update(counts);
    
//     auto posterior_alphas = cd.getAlphas();
//     std::vector<double> expected = {6.0, 4.0, 3.0};
//     EXPECT_TRUE(vector_approx_equal(posterior_alphas, expected));
// }

// TEST_F(ConjugateCategoricalDirichletUpdateTest, PosteriorMean) {
//     std::vector<int> counts = {5, 3, 2};
//     cd.update(counts);
    
//     auto mean = cd.getPosteriorMean();
//     double sum = std::accumulate(mean.begin(), mean.end(), 0.0);
//     EXPECT_TRUE(approx_equal(sum, 1.0));
    
//     // Posterior mean should be proportional to prior + counts
//     // E[p] = (alpha + counts) / (sum(alpha) + sum(counts))
//     EXPECT_TRUE(approx_equal(mean[0], 6.0 / 10.0));
// }

// TEST_F(ConjugateCategoricalDirichletUpdateTest, LogPredictiveProb) {
//     std::vector<int> counts = {5, 0, 0};
//     cd.update(counts);
    
//     double log_prob = cd.logPredictiveProb(0);
//     EXPECT_TRUE(std::isfinite(log_prob));
//     EXPECT_LT(log_prob, 0);  // log probability should be negative
// }

// Test suite for sampling
class ConjugateCategoricalDirichletSamplingTest : public ::testing::Test {
protected:
    ConjugateCategoricalDirichlet cd{std::vector<double>{1.0, 1.0, 1.0}};
};

// TEST_F(ConjugateCategoricalDirichletSamplingTest, SampleReturnsValidDistribution) {
//     std::vector<double> sample = cd.sample();
    
//     EXPECT_EQ(sample.size(), 3);
//     double sum = std::accumulate(sample.begin(), sample.end(), 0.0);
//     EXPECT_TRUE(approx_equal(sum, 1.0));
    
//     for (double val : sample) {
//         EXPECT_GE(val, 0.0);
//         EXPECT_LE(val, 1.0);
//     }
// }