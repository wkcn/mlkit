#pragma once

#include <boosted_tree/array.h>
#include <boosted_tree/csr_matrix.h>
#include <gtest/gtest.h>

#include <array>
#include <initializer_list>

TEST(TestArray, test_op) {
  Array<float, 3> a{1, 2, 3};
  Array<float, 3> b{4, 5, 6};
  std::array<float, 3> target;
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
