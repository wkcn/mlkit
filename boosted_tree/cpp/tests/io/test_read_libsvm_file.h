#pragma once

#include <vector>

#include <gtest/gtest.h>
#include <boosted_tree/csr_matrix.h>
#include <boosted_tree/io.h>

TEST(TestIO, ReadLibSVMFile) {
  auto [smat, labels] = ReadLibSVMFile<float, int>("./tests/io/simple_data.txt");
  ASSERT_EQ(labels, (std::vector<int>{1, 2, 4}));
  ASSERT_EQ(smat[0], (std::vector<float>{0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 7, 0}));
  ASSERT_EQ(smat[1], (std::vector<float>{0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0}));
  ASSERT_EQ(smat[2], (std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 12}));
}