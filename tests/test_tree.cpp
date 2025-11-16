#include <gtest/gtest.h>
#include "bayes_tree/bayes_tree.hpp"

TEST(BayesTreeTest, BasicTreePredictionTest) {
    BayesTree tree;
    double result = tree.predict(2.0);
    EXPECT_EQ(result, 5.0);
}