#ifndef BOOSTED_TREE_CSR_MATRIX_H_
#define BOOSTED_TREE_CSR_MATRIX_H_

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "./matrix.h"
#include "./logging.h"

typedef int64_t dim_t;

template<typename T>
class CSRMatrix {
public:
  CSRMatrix(dim_t rows, dim_t cols);
  void reset(const std::vector<dim_t> &row,
             const std::vector<dim_t> &col,
             const std::vector<T> &data);
  void set(const std::vector<dim_t> &row,
           const std::vector<dim_t> &col,
           const std::vector<T> &data);
  void set(dim_t row, dim_t col, const T &data);
  Matrix<T> todense();

private:
  dim_t rows_, cols_;
  std::vector<dim_t> offsets_;  // row offsets
  std::vector<dim_t> indices_;  // row indices
  std::vector<T> values_;
};

template<typename T>
CSRMatrix<T>::CSRMatrix(dim_t rows, dim_t cols) {
  CHECK_GE(rows, 0);
  CHECK_GE(cols, 0);
  rows_ = rows;
  cols_ = cols;
  offsets_.resize(rows_ + 1, 0);
}

template<typename T>
void CSRMatrix<T>::reset(const std::vector<dim_t> &row,
                         const std::vector<dim_t> &col,
                         const std::vector<T> &data) {
  std::vector<std::vector<std::pair<dim_t, T> > > rec(rows_);
  CHECK_EQ(row.size(), col.size());
  CHECK_EQ(row.size(), data.size());
  indices_.clear();
  values_.clear();
  const dim_t n = row.size();
  for (dim_t i = 0; i < n; ++i) {
    rec[row[i]].push_back({col[i], data[i]});
  }
  dim_t last_offset = 0;
  for (dim_t r = 0; r < rows_; ++r) {
    auto &vs = rec[r];
    offsets_[r] = last_offset; // the begin element of this row
    if (!vs.empty()) {
      // offsets_[r]:offsets_[r+1]
      sort(vs.begin(), vs.end());
      for (auto &p : vs) {
        indices_.push_back(p.first);
        values_.push_back(p.second);
      }
      last_offset += vs.size();
    }
  }
  TEST_EQ(last_offset, values_.size());
  offsets_.back() = last_offset;
}

template<typename T>
void CSRMatrix<T>::set(const std::vector<dim_t> &row,
                       const std::vector<dim_t> &col,
                       const std::vector<T> &data) {
}

template<typename T>
void CSRMatrix<T>::set(dim_t row, dim_t col, const T &data) {
  set(std::vector<dim_t>{row},
      std::vector<dim_t>{col},
      std::vector<T>{data});
}

template<typename T>
Matrix<T> CSRMatrix<T>::todense() {
  Matrix<T> mat(rows_, cols_, 0);
  for (dim_t r = 0; r < rows_; ++r) {
    if (offsets_[r] != -1) {
      for (dim_t i = offsets_[r]; i < offsets_[r+1]; ++i) {
        mat.data[r][indices_[i]] = values_[i];
      } 
    }
  }
  return mat;
}

#endif
