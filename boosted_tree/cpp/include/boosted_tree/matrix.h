#ifndef BOOSTED_TREE_MATRIX_H_
#define BOOSTED_TREE_MATRIX_H_

#include <initializer_list>
#include <iostream>
#include <memory>
#include <vector>

#include "./logging.h"

typedef int64_t dim_t;

template<typename T>
class DenseRow;

template<typename T>
struct DenseChunk {
  std::vector<std::vector<T> > data;
};

template<typename T>
class Matrix {
public:
  Matrix(dim_t rows, dim_t cols);
  Matrix(dim_t rows, dim_t cols, T val);
public:
  DenseRow<T> operator[](dim_t row);
  dim_t length() const;
private:
  std::shared_ptr<DenseChunk<T> > data_;
  friend DenseRow<T>;
};

template<typename T>
class DenseRow {
public:
  DenseRow(std::shared_ptr<DenseChunk<T> > data, dim_t row);
  T& operator[](dim_t col);
  dim_t length() const;
public:
  template <typename U, typename VT>
  friend bool operator==(const DenseRow<U> &a, const VT &b);
private:
  std::shared_ptr<DenseChunk<T> > data_;
  dim_t row_;
}; 

template<typename T, typename VT>
bool operator==(const DenseRow<T> &a, const VT &b) {
  if (a.length() != b.size()) return false;
  auto rdata = a.data_->data[a.row_];
  for (auto pa = rdata.begin(), pb = b.begin(); pb != b.end(); ++pa, ++pb) {
    if (*pa != *pb) return false;
  }
  return true;
}

template<typename T>
Matrix<T>::Matrix(dim_t rows, dim_t cols) {
  CHECK_GE(rows, 0);
  CHECK_GE(cols, 0);
  data_.reset(new DenseChunk<T>);
  std::vector<T> tmp(cols);
  data_->data.resize(rows, tmp);
}

template<typename T>
Matrix<T>::Matrix(dim_t rows, dim_t cols, T val) {
  CHECK_GE(rows, 0);
  CHECK_GE(cols, 0);
  data_.reset(new DenseChunk<T>);
  std::vector<T> tmp(cols, val);
  data_->data.resize(rows, tmp);
}

template<typename T>
DenseRow<T> Matrix<T>::operator[](dim_t row) {
  return DenseRow<T>(data_, row);
}

template<typename T>
dim_t Matrix<T>::length() const {
  return data_->data.size();
}

template<typename T>
DenseRow<T>::DenseRow(
    std::shared_ptr<DenseChunk<T> > data, dim_t row) :
    data_(data), row_(row) {
}

template<typename T>
T& DenseRow<T>::operator[](dim_t col) {
  return data_->data[row_][col];
}

template<typename T>
dim_t DenseRow<T>::length() const{
  return data_->data[row_].size();
}

template<typename T>
std::ostream& operator<<(std::ostream &os, const Matrix<T> &mat) {
  const size_t row = mat.length();
  if (row == 0) return os;
  const size_t col = mat[0].length();
  if (col == 0) return os;
  for (int r = 0; r < row; ++r) {
    bool first = true;
    for (int c = 0; c < col; ++c) {
      if (!first)
        os << " ";
      else
        first = false;
      os << mat[r][c];
    }
    os << std::endl;
  }
  return os;
}

#endif
