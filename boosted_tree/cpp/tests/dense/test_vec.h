#pragma once

#include <boosted_tree/csr_matrix.h>
#include <boosted_tree/vec.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

TEST(TestVec, test_op) {
  Vec<float> a{1, 2, 3};
  Vec<float> b{4, 5, 6};
  std::vector<float> target(3);
  for (int i = 0; i < 3; ++i) {
    target[i] = a[i] + b[i];
  }
  ASSERT_EQ(a + b, target);

  for (int i = 0; i < 3; ++i) {
    target[i] = a[i] - b[i];
  }
  ASSERT_EQ(a - b, target);

  for (int i = 0; i < 3; ++i) {
    target[i] = a[i] * b[i];
  }
  ASSERT_EQ(a * b, target);

  for (int i = 0; i < 3; ++i) {
    target[i] = a[i] / b[i];
  }
  ASSERT_EQ(a / b, target);
}
