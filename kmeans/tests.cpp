#include <gtest/gtest.h>
#include "vec.h"

TEST(TEST_VEC_OP_VEC, OP_VEC) {
  const int NDIM = 100;
  Vec<int> a(NDIM);
  Vec<int> b(NDIM);
  for (int i = 0; i < NDIM; ++i) {
    a[i] = rand();
  }
  for (int i = 0; i < NDIM; ++i) {
    b[i] = rand();
  }
  Vec<int> c(NDIM);
  for (int i = 0; i < NDIM; ++i) {
    c[i] = a[i] + b[i];
  }
  EXPECT_EQ(a + b, c);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
