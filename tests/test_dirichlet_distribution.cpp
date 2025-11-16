// test_dirichlet.cpp
#include <gtest/gtest.h>
#include "bayes_tree/dirichlet_distribution.hpp"
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

// Test suite for constructor
class DirichletConstructorTest : public ::testing::Test {};

TEST_F(DirichletConstructorTest, ValidAlpha) {
    std::vector<double> alpha = {1.0, 2.0, 3.0};
    EXPECT_NO_THROW(DirichletDistribution d(alpha));
}

TEST_F(DirichletConstructorTest, RejectsEmptyAlpha) {
    std::vector<double> alpha = {};
    EXPECT_THROW(DirichletDistribution d(alpha), std::invalid_argument);
}

TEST_F(DirichletConstructorTest, RejectsNegativeAlpha) {
    std::vector<double> alpha = {1.0, -2.0, 3.0};
    EXPECT_THROW(DirichletDistribution d(alpha), std::invalid_argument);
}

TEST_F(DirichletConstructorTest, RejectsZeroAlpha) {
    std::vector<double> alpha = {1.0, 0.0, 3.0};
    EXPECT_THROW(DirichletDistribution d(alpha), std::invalid_argument);
}

// Test suite for dimension
class DirichletDimensionTest : public ::testing::Test {
protected:
    std::vector<double> alpha = {1.0, 2.0, 3.0, 4.0};
    DirichletDistribution d{alpha};
};

TEST_F(DirichletDimensionTest, DimensionReturnsCorrectSize) {
    EXPECT_EQ(d.dimension(), 4);
}

// Test suite for mean
class DirichletMeanTest : public ::testing::Test {};

TEST_F(DirichletMeanTest, MeanOfUniformDistribution) {
    std::vector<double> alpha = {1.0, 1.0, 1.0};
    DirichletDistribution d(alpha);
    auto mean = d.mean();
    std::vector<double> expected = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    EXPECT_TRUE(vector_approx_equal(mean, expected));
}

TEST_F(DirichletMeanTest, MeanOfNonUniformDistribution) {
    std::vector<double> alpha = {2.0, 3.0, 5.0};
    DirichletDistribution d(alpha);
    auto mean = d.mean();
    std::vector<double> expected = {0.2, 0.3, 0.5};
    EXPECT_TRUE(vector_approx_equal(mean, expected));
}

// Test suite for variance
class DirichletVarianceTest : public ::testing::Test {
protected:
    std::vector<double> alpha = {2.0, 3.0, 5.0};
    DirichletDistribution d{alpha};
};

TEST_F(DirichletVarianceTest, VarianceCalculation) {
    auto var = d.variance();
    double alpha_sum = 10.0;
    std::vector<double> expected = {
        (2.0 * (alpha_sum-2.0)) / (alpha_sum * alpha_sum * (alpha_sum + 1.0)),
        (3.0 * (alpha_sum-3.0)) / (alpha_sum * alpha_sum * (alpha_sum + 1.0)),
        (5.0 * (alpha_sum-5.0)) / (alpha_sum * alpha_sum * (alpha_sum + 1.0))
    };
    EXPECT_TRUE(vector_approx_equal(var, expected));
}

// Test suite for sampling
class DirichletSamplingTest : public ::testing::Test {
protected:
    std::vector<double> alpha = {2.0, 3.0, 5.0};
    DirichletDistribution d{alpha};
};

TEST_F(DirichletSamplingTest, SampleHasCorrectDimension) {
    auto sample = d.sample();
    EXPECT_EQ(sample.size(), 3);
}

TEST_F(DirichletSamplingTest, SampleSumsToOne) {
    auto sample = d.sample();
    double sum = std::accumulate(sample.begin(), sample.end(), 0.0);
    EXPECT_TRUE(approx_equal(sum, 1.0));
}

TEST_F(DirichletSamplingTest, SampleValuesInRange) {
    auto sample = d.sample();
    for (double val : sample) {
        EXPECT_GT(val, 0.0);
        EXPECT_LT(val, 1.0);
    }
}

TEST_F(DirichletSamplingTest, MultipleSamplesReturnCorrectCount) {
    auto samples = d.sample(10);
    EXPECT_EQ(samples.size(), 10);
    EXPECT_EQ(samples[0].size(), 3);
}

// Test suite for sample statistics
class DirichletSampleStatisticsTest : public ::testing::Test {};

TEST_F(DirichletSampleStatisticsTest, EmpiricalMeanConverges) {
    std::vector<double> alpha = {5.0, 5.0, 5.0};
    DirichletDistribution d(alpha);
    
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
    EXPECT_TRUE(vector_approx_equal(empirical_mean, theoretical_mean, 0.01));
}

// Test suite for log PDF
class DirichletLogPdfTest : public ::testing::Test {
protected:
    std::vector<double> alpha = {2.0, 3.0, 5.0};
    DirichletDistribution d{alpha};
};

TEST_F(DirichletLogPdfTest, ComputesForValidPoint) {
    std::vector<double> x = {0.2, 0.3, 0.5};
    double logp = d.logPdf(x);
    EXPECT_TRUE(std::isfinite(logp));
}

TEST_F(DirichletLogPdfTest, RejectsPointNotSummingToOne) {
    std::vector<double> x = {0.2, 0.3, 0.4};
    EXPECT_THROW(d.logPdf(x), std::invalid_argument);
}

TEST_F(DirichletLogPdfTest, RejectsWrongDimension) {
    std::vector<double> x = {0.5, 0.5};
    EXPECT_THROW(d.logPdf(x), std::invalid_argument);
}

TEST_F(DirichletLogPdfTest, ReturnsNegativeInfinityForBoundaryPoint) {
    std::vector<double> x = {0.0, 0.5, 0.5};
    double logp = d.logPdf(x);
    EXPECT_TRUE(std::isinf(logp));
    EXPECT_LT(logp, 0);
}

TEST_F(DirichletLogPdfTest, LogPdfHigherAtMeanThanExtreme) {
    auto mean_point = d.mean();
    std::vector<double> extreme_point = {0.01, 0.01, 0.98};
    EXPECT_GT(d.logPdf(mean_point), d.logPdf(extreme_point));
}

// Test suite for get/set alpha
class DirichletAlphaTest : public ::testing::Test {
protected:
    std::vector<double> alpha = {2.0, 3.0, 5.0};
    DirichletDistribution d{alpha};
};

TEST_F(DirichletAlphaTest, GetAlphaReturnsCorrectValues) {
    auto retrieved = d.getAlpha();
    EXPECT_TRUE(vector_approx_equal(retrieved, alpha));
}

TEST_F(DirichletAlphaTest, SetAlphaWithValidValues) {
    std::vector<double> new_alpha = {1.0, 2.0, 3.0};
    d.setAlpha(new_alpha);
    EXPECT_TRUE(vector_approx_equal(d.getAlpha(), new_alpha));
}

TEST_F(DirichletAlphaTest, SetAlphaRejectsWrongSize) {
    std::vector<double> wrong_size = {1.0, 2.0};
    EXPECT_THROW(d.setAlpha(wrong_size), std::invalid_argument);
}

TEST_F(DirichletAlphaTest, SetAlphaRejectsNegativeValues) {
    std::vector<double> invalid = {1.0, -2.0, 3.0};
    EXPECT_THROW(d.setAlpha(invalid), std::invalid_argument);
}