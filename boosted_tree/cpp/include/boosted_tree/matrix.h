#ifndef BOOSTED_TREE_MATRIX_H_
#define BOOSTED_TREE_MATRIX_H_

#include <iostream>
#include <vector>

#include "./logging.h"

typedef int64_t dim_t;

template<typename T>
class Matrix {
public:
  Matrix(dim_t rows, dim_t cols);
  Matrix(dim_t rows, dim_t cols, T val);
  std::vector<std::vector<T> > data;
private:
  dim_t rows_, cols_;
};


template<typename T>
Matrix<T>::Matrix(dim_t rows, dim_t cols) {
  CHECK_GE(rows, 0);
  CHECK_GE(cols, 0);
  rows_ = rows;
  cols_ = cols;
  std::vector<T> tmp(cols_);
  data.resize(rows_, tmp);
}


template<typename T>
Matrix<T>::Matrix(dim_t rows, dim_t cols, T val) {
  CHECK_GE(rows, 0);
  CHECK_GE(cols, 0);
  rows_ = rows;
  cols_ = cols;
  std::vector<T> tmp(cols_, val);
  data.resize(rows_, tmp);
}


template<typename T>
std::ostream& operator<<(std::ostream &os, const Matrix<T> &mat) {
  const std::vector<std::vector<T> > &data = mat.data;
  const size_t row = data.size();
  if (row == 0) return os;
  const size_t col = data[0].size();
  if (col == 0) return os;
  for (int r = 0; r < row; ++r) {
    bool first = true;
    for (int c = 0; c < col; ++c) {
      if (!first)
        os << " ";
      else
        first = false;
      os << data[r][c];
    }
    os << std::endl;
  }
  return os;
}

#endif
