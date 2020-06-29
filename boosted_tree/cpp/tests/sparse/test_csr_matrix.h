#pragma once
#include <boosted_tree/csr_matrix.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <valarray>
#include <vector>

TEST(TestCSRMatrix, todense) {
  std::vector<dim_t> row{0, 0, 1, 2, 2, 2};
  std::vector<dim_t> col{0, 2, 2, 0, 1, 2};
  std::vector<int> data{1, 2, 3, 4, 5, 6};
  CSRMatrix<int> smat(3, 3);
  smat.reset(row, col, data);
  Matrix<int> mat = smat.todense();
  ASSERT_EQ(mat[0], (std::vector<int>{1, 0, 2}));
  ASSERT_EQ(mat[1], (std::vector<int>{0, 0, 3}));
  ASSERT_EQ(mat[2], (std::vector<int>{4, 5, 6}));
}

TEST(TestCSRMatrix, getitem) {
  std::vector<std::vector<int>> mat{{1, 0, 2}, {0, 0, 3}, {4, 5, 6}};
  std::vector<dim_t> row;
  std::vector<dim_t> col;
  std::vector<int> data;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      if (mat[r][c]) {
        row.push_back(r);
        col.push_back(c);
        data.push_back(mat[r][c]);
      }
    }
  }
  CSRMatrix<int> smat(3, 3);
  smat.reset(row, col, data);

  for (int r = 0; r < 3; ++r) {
    ASSERT_EQ(smat[r], mat[r]);
    for (int c = 0; c < 3; ++c) {
      ASSERT_EQ(smat[r][c], mat[r][c]);
    }
  }

  std::vector<std::vector<int>> orders = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                          {1, 2, 0}, {2, 0, 1}, {2, 1, 0},
                                          {0, 1, 1}, {1, 0, 1}, {2, 2, 1}};
  for (int r = 0; r < 3; ++r) {
    for (auto &order : orders) {
      std::valarray<int> left = smat[r].at(order.begin(), order.end());
      std::valarray<int> right(3);
      for (int j = 0; j < 3; ++j) {
        right[j] = mat[r][order[j]];
      }
      ASSERT_EQ((left == right).min(), 1);
    }
  }
}

TEST(TestCSRMatrix, transpose) {
  std::vector<std::vector<int>> mat{{1, 0, 2}, {0, 0, 3}, {4, 5, 6}, {7, 4, 0}};
  std::vector<dim_t> row;
  std::vector<dim_t> col;
  std::vector<int> data;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      if (mat[r][c]) {
        row.push_back(r);
        col.push_back(c);
        data.push_back(mat[r][c]);
      }
    }
  }
  CSRMatrix<int> smat(4, 3);
  smat.reset(row, col, data);
  CSRMatrix<int> smatT = smat.transpose();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      ASSERT_EQ(smat[r][c], smatT[c][r]);
    }
  }
}

TEST(TestCSRMatrix, setitem) {
  std::vector<std::vector<int>> mat{{1, 0, 2}, {0, 0, 3}, {4, 5, 6}};
  std::vector<dim_t> row;
  std::vector<dim_t> col;
  std::vector<int> data;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      if (mat[r][c]) {
        row.push_back(r);
        col.push_back(c);
        data.push_back(mat[r][c]);
      }
    }
  }
  CSRMatrix<int> smat(3, 3);
  smat.reset(row, col, data);

  std::vector<std::vector<int>> new_mat{{5, 3, 2}, {0, 4, 0}, {7, 0, 6}};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      smat[r].set(c, new_mat[r][c]);
    }
  }
  for (int r = 0; r < 3; ++r) {
    ASSERT_EQ(smat[r], new_mat[r]);
  }
  smat.compress();
  for (int r = 0; r < 3; ++r) {
    ASSERT_EQ(smat[r], new_mat[r]);
  }
}

TEST(TestCSRRow, todense) {
  std::vector<std::vector<int>> mat{{1, 0, 2}, {0, 0, 3}, {4, 5, 6}};
  std::vector<dim_t> row;
  std::vector<dim_t> col;
  std::vector<int> data;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      if (mat[r][c]) {
        row.push_back(r);
        col.push_back(c);
        data.push_back(mat[r][c]);
      }
    }
  }
  CSRMatrix<int> smat(3, 3);
  smat.reset(row, col, data);
  for (int r = 0; r < 3; ++r) {
    ASSERT_EQ(smat[r].todense(), Vec<int>(mat[r].begin(), mat[r].end()));
  }
}
