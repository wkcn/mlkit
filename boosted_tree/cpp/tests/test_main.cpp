#include <gtest/gtest.h>

#include "./dense/dense.h"
#include "./io/io.h"
#include "./sparse/sparse.h"
#include "./quantile/quantile.h"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
