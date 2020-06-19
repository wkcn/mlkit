#ifndef BOOSTED_TREE_CSR_MATRIX_H_
#define BOOSTED_TREE_CSR_MATRIX_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "./coo_matrix.h"
#include "./matrix.h"
#include "./logging.h"
#include "./vec.h"

typedef int64_t dim_t;

template<typename T>
class CSRRow;

template<typename T>
struct CSRChunk {
  std::vector<dim_t> offsets;  // row offsets
  std::vector<dim_t> indices;  // row indices
  std::vector<T> values;
};

template<typename T>
class CSRMatrix {
public:
  CSRMatrix() = default;
  CSRMatrix(const CSRMatrix&) = default;
  CSRMatrix(CSRMatrix&&) = default;
  CSRMatrix(dim_t rows, dim_t cols);
  CSRMatrix& operator=(const CSRMatrix&) = default;
  CSRMatrix& operator=(CSRMatrix&&) = default;

  void reset(const COOMatrix<T> &smat);
  void reset(const std::vector<dim_t> &row,
             const std::vector<dim_t> &col,
             const std::vector<T> &data);
  void compress();
  Matrix<T> todense() const;
  COOMatrix<T> tocoo() const;
  CSRMatrix<T> transpose() const;
public:
  CSRRow<T> operator[](dim_t row) const;
  dim_t length() const;
private:
  std::shared_ptr<CSRChunk<T> > data_;
  dim_t rows_, cols_;
  friend CSRRow<T>;
};

template<typename T>
class CSRRow {
public:
  CSRRow(std::shared_ptr<CSRChunk<T> > data, dim_t row, dim_t cols);
  T operator[](dim_t col) const;
  template <typename Iterator>
  Vec<T> at(Iterator first, Iterator last) const;
  void set(dim_t col, const T &value);
  Vec<T> todense() const;
  dim_t length() const;
public:
  template <typename U, typename VT>
  friend bool operator==(const CSRRow<U> &a, const VT &b);
private:
  std::shared_ptr<CSRChunk<T> > data_;
  dim_t row_;
  dim_t cols_;
}; 

template<typename T, typename VT>
bool operator==(const CSRRow<T> &a, const VT &b) {
  auto &offsets = a.data_->offsets;
  auto &indices = a.data_->indices;
  auto &values = a.data_->values;
  dim_t offset = offsets[a.row_];
  const dim_t offset_end = offsets[a.row_ + 1];
  for (dim_t i = 0; i < b.size(); ++i) {
    if (offset >= offset_end || indices[offset] != i) {
      if (b[i] != 0) return false;
    } else {
      if (b[i] != values[offset]) return false;
      ++offset;
    }
  }
  return true;
}

template<typename T>
CSRMatrix<T>::CSRMatrix(dim_t rows, dim_t cols) {
  CHECK_GE(rows, 0);
  CHECK_GE(cols, 0);
  rows_ = rows;
  cols_ = cols;
  data_.reset(new CSRChunk<T>);
  data_->offsets.resize(rows_ + 1, 0);
}

template<typename T>
CSRRow<T> CSRMatrix<T>::operator[](dim_t row) const {
  return CSRRow<T>(data_, row, cols_);
}

template<typename T>
dim_t CSRMatrix<T>::length() const {
  return rows_;
}

template <typename T>
void CSRMatrix<T>::reset(const COOMatrix<T> &smat) {
  const COOChunk<T> &chunk = smat.data();
  reset(chunk.row, chunk.col, chunk.data);
}

template<typename T>
void CSRMatrix<T>::reset(const std::vector<dim_t> &row,
                         const std::vector<dim_t> &col,
                         const std::vector<T> &data) {
  std::vector<std::vector<std::pair<dim_t, T> > > rec(rows_);
  CHECK_EQ(row.size(), col.size());
  CHECK_EQ(row.size(), data.size());

  data_.reset(new CSRChunk<T>);
  data_->offsets.resize(rows_ + 1, 0);

  auto &offsets = data_->offsets;
  auto &indices = data_->indices;
  auto &values = data_->values;

  indices.clear();
  values.clear();
  const dim_t n = row.size();
  for (dim_t i = 0; i < n; ++i) {
    rec[row[i]].push_back({col[i], data[i]});
  }
  dim_t last_offset = 0;

  for (dim_t r = 0; r < rows_; ++r) {
    auto &vs = rec[r];
    offsets[r] = last_offset; // the begin element of this row
    if (!vs.empty()) {
      // offsets[r]:offsets[r+1]
      sort(vs.begin(), vs.end());
      for (auto &p : vs) {
        indices.push_back(p.first);
        values.push_back(p.second);
      }
      last_offset += vs.size();
    }
  }
  TEST_EQ(last_offset, values.size());
  offsets.back() = last_offset;
}

template<typename T>
void CSRMatrix<T>::compress() {
  auto &offsets = data_->offsets;
  auto &indices = data_->indices;
  auto &values = data_->values;
  dim_t vi = 0;
  for (dim_t r = 0; r < rows_; ++r) {
    const dim_t offset_begin = offsets[r];
    const dim_t offset_end = offsets[r + 1];
    offsets[r] = vi;
    for (dim_t t = offset_begin; t < offset_end; ++t) {
      if (values[t] != 0) {
        values[vi] = values[t];
        indices[vi] = indices[t];
        ++vi;
      }
    }
  }
  offsets[rows_] = vi;
  indices.resize(vi);
  values.resize(vi);
}

template<typename T>
Matrix<T> CSRMatrix<T>::todense() const {
  Matrix<T> mat(rows_, cols_, 0);
  auto &offsets = data_->offsets;
  auto &indices = data_->indices;
  auto &values = data_->values;
  for (dim_t r = 0; r < rows_; ++r) {
    if (offsets[r] != -1) {
      for (dim_t i = offsets[r]; i < offsets[r+1]; ++i) {
        mat[r][indices[i]] = values[i];
      } 
    }
  }
  return mat;
}

template<typename T>
COOMatrix<T> CSRMatrix<T>::tocoo() const {
  COOMatrix<T> smat(rows_, cols_);
  auto &offsets = data_->offsets;
  auto &indices = data_->indices;
  auto &values = data_->values;
  COOChunk<T> &chunk = smat.data();
  for (dim_t r = 0; r < rows_; ++r) {
    if (offsets[r] != -1) {
      for (dim_t i = offsets[r]; i < offsets[r+1]; ++i) {
        chunk.row.push_back(r);
        chunk.col.push_back(indices[i]);
        chunk.data.push_back(values[i]);
      }
    }
  }
  return smat;
}

template<typename T>
CSRMatrix<T> CSRMatrix<T>::transpose() const {
  COOMatrix<T> smat = tocoo();
  COOChunk<T> &chunk = smat.data();
  CSRMatrix<T> res(cols_, rows_);
  res.reset(chunk.col, chunk.row, chunk.data);
  return res;
}

template<typename T>
CSRRow<T>::CSRRow(
    std::shared_ptr<CSRChunk<T> > data, dim_t row, dim_t cols) :
    data_(data), row_(row), cols_(cols) {
}

template<typename T>
T CSRRow<T>::operator[](dim_t col) const {
  auto &offsets = data_->offsets;
  auto &indices = data_->indices;
  auto &values = data_->values;
  const dim_t offset_begin = offsets[row_];
  const dim_t offset_end = offsets[row_ + 1];
  const auto rindices_begin = indices.begin() + offset_begin;
  const auto rindices_end = indices.begin() + offset_end;
  auto p = lower_bound(rindices_begin, rindices_end, col); 
  return (p != rindices_end && *p == col) ? values[p - indices.begin()] : 0;
}

template <typename T>
template <typename Iterator>
Vec<T> CSRRow<T>::at(Iterator first, Iterator last) const {
  std::vector<dim_t> cols(first, last);
  auto &offsets = data_->offsets;
  auto &indices = data_->indices;
  auto &values = data_->values;
  const dim_t offset_begin = offsets[row_];
  const dim_t offset_end = offsets[row_ + 1];


  const int n = cols.size();
  std::vector<dim_t> inds(n);
  iota(inds.begin(), inds.end(), 0);
  std::sort(inds.begin(), inds.end(), [&cols](const int a, const int b) {
      return cols[a] < cols[b];
  });
  Vec<T> out(n);
  auto p = indices.begin() + offset_begin;
  const auto rindices_end = indices.begin() + offset_end;
  for (dim_t i = 0; i < n; ++i) {
    dim_t t = inds[i];
    dim_t col = cols[t];
    p = lower_bound(p, rindices_end, col);
    out[t] = (p != rindices_end && *p == col) ? values[p - indices.begin()] : 0;
  }
  return out;
}

template <typename T>
void CSRRow<T>::set(dim_t col, const T &value) {
  auto &offsets = data_->offsets;
  auto &indices = data_->indices;
  auto &values = data_->values;
  const dim_t offset_begin = offsets[row_];
  const dim_t offset_end = offsets[row_ + 1];
  const auto rindices_begin = indices.begin() + offset_begin;
  const auto rindices_end = indices.begin() + offset_end;
  auto p = lower_bound(rindices_begin, rindices_end, col); 
  dim_t offset = p - indices.begin();
  if (p != rindices_end && *p == col) {
    values[offset] = value;
  } else if (value != 0) {
    bool first = p == rindices_begin;
    indices.insert(p, col);
    values.insert(values.begin() + offset, value);
    if (first) {
      offsets[row_] = offset;
    }
    for (dim_t r = row_ + 1; r < offsets.size(); ++r) {
      ++offsets[r];
    } 
  }
}

template<typename T>
Vec<T> CSRRow<T>::todense() const {
  auto &offsets = data_->offsets;
  auto &indices = data_->indices;
  auto &values = data_->values;
  dim_t offset = offsets[row_];
  const dim_t offset_end = offsets[row_ + 1];
  Vec<T> vec(cols_);
  for (dim_t i = 0; i < cols_; ++i) {
    if (offset >= offset_end || indices[offset] != i) {
      vec[i] = 0;
    } else {
      vec[i] = values[offset];
      ++offset;
    }
  }
  return vec;
}

template<typename T>
dim_t CSRRow<T>::length() const {
  return cols_;
}

#endif
