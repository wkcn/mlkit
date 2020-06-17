#pragma once
#include <vector>

#include <gtest/gtest.h>
#include <boosted_tree/csr_matrix.h>

TEST(TestCSRMatrix, todense) {
  std::vector<dim_t> row{0, 0, 1, 2, 2, 2};
  std::vector<dim_t> col{0, 2, 2, 0, 1, 2};
  std::vector<int> data{1, 2, 3, 4, 5, 6};
  CSRMatrix<int> smat(3, 3);
  smat.reset(row, col, data);
  Matrix<int> mat = smat.todense();
  ASSERT_EQ(mat.data[0], (std::vector<int>{1, 0, 2}));
  ASSERT_EQ(mat.data[1], (std::vector<int>{0, 0, 3}));
  ASSERT_EQ(mat.data[2], (std::vector<int>{4, 5, 6}));
}
