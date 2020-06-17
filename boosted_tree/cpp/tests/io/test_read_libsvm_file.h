#pragma once

#include <vector>

#include <gtest/gtest.h>
#include <boosted_tree/csr_matrix.h>
#include <boosted_tree/io.h>

TEST(TestIO, ReadLibSVMFile) {
  auto [smat, labels] = ReadLibSVMFile("./tests/io/simple_data.txt");
  ASSERT_EQ(labels, (std::vector<float>{1, 2, 4}));
} 
