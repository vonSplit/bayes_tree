#include <gtest/gtest.h>
#include <cmath>
#include "bayes_tree/categorical_distribution.hpp"

// Test suite for CategoricalDistribution initialization
class CategoricalDistributionTest : public ::testing::Test {
protected:
    std::vector<double> probs = {0.2, 0.3, 0.5};
    CategoricalDistribution dist{probs};
};

TEST_F(CategoricalDistributionTest, BasicInitialisationTest) {
    EXPECT_EQ(dist.probs().size(), probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        EXPECT_NEAR(dist.probs()[i], probs[i], 1e-9);
    }
    double a=4;
    EXPECT_NEAR(a, 4.0, 1e-9);
}

TEST_F(CategoricalDistributionTest, CheckNormalisedProbs) {
    double sum = 0.0;
    for (double p : dist.probs()) {
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

// Test suite for log-likelihood calculations
class CategoricalDistributionLogLikelihoodTest : public ::testing::Test {
protected:
    std::vector<double> probs = {0.2, 0.3, 0.5};
    CategoricalDistribution dist{probs};
};

TEST_F(CategoricalDistributionLogLikelihoodTest, LogLikelihoodWithKnownResult) {
    // Counts = [2, 1, 1] → loglike = 2*log(0.2) + 1*log(0.3) + 1*log(0.5)
    std::vector<int> counts = {2, 1, 1};
    double ll = dist.log_likelihood(counts);
    double expected = 2 * std::log(0.2) + std::log(0.3) + std::log(0.5);
    EXPECT_NEAR(ll, expected, 1e-9);
}

TEST_F(CategoricalDistributionLogLikelihoodTest, EdgeCaseZeroCount) {
    std::vector<int> counts = {0, 0, 4};
    double ll = dist.log_likelihood(counts);
    double expected = 4 * std::log(0.5);
    EXPECT_NEAR(ll, expected, 1e-9);
}

TEST_F(CategoricalDistributionLogLikelihoodTest, ImpossibleEventZeroProbabilityWithNonzeroCount) {
    std::vector<double> probs_zero = {0.0, 0.5, 0.5};
    CategoricalDistribution dist_zero(probs_zero);
    std::vector<int> bad_counts = {1, 0, 0};
    double ll_bad = dist_zero.log_likelihood(bad_counts);
    
    EXPECT_TRUE(std::isinf(ll_bad));
    EXPECT_LT(ll_bad, 0);
}


// 
// #include <cassert>
// #include <cmath>


// int main() {
//      std::cout << "JT: Debugging categorical_distribution" << std::endl;

//     // 1. Basic initialisation test
//     std::vector<double> probs = {0.2, 0.3, 0.5};
//     CategoricalDistribution dist(probs);
//     //assert(false);

//    // Check probabilities normalised
//     double sum = 0.0;
//     for (double p : dist.probs()) sum += p;
//     assert(std::fabs(sum - 1.0) < 1e-9);

//     // 2. Log-likelihood test with known result
//     //Counts = [2, 1, 1] → loglike = 2*log(0.2) + 1*log(0.3) + 1*log(0.5)
//     std::vector<int> counts = {2, 1, 1};
//     double ll = dist.log_likelihood(counts);
//     double expected = 2 * std::log(0.2) + std::log(0.3) + std::log(0.5);
//     assert(std::fabs(ll - expected) < 1e-9);

//     // // 3. Edge case: zero count shouldn’t affect log-likelihood
//     std::vector<int> counts2 = {0, 0, 4};
//     double ll2 = dist.log_likelihood(counts2);
//     double expected2 = 4 * std::log(0.5);
//     assert(std::fabs(ll2 - expected2) < 1e-9);

//     // // 4. Impossible event: zero-probability category with nonzero count
//     std::vector<double> probs_zero = {0.0, 0.5, 0.5};
//     CategoricalDistribution dist_zero(probs_zero);
//     std::vector<int> bad_counts = {1, 0, 0};
//     double ll_bad = dist_zero.log_likelihood(bad_counts);
//     assert(std::isinf(ll_bad) && ll_bad < 0);  // should be -inf

//     std::cout << "All tests passed.\n";
//     return 0;
// }
