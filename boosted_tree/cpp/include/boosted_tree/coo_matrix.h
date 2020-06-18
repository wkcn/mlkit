#ifndef BOOSTED_TREE_COO_MATRIX_H_
#define BOOSTED_TREE_COO_MATRIX_H_

#include <cstddef>
#include <vector>
#include <memory>

#include "./logging.h"

typedef int64_t dim_t;

template<typename T>
class COORow;

template <typename T>
struct COOChunk {
  std::vector<dim_t> row;
  std::vector<dim_t> col;
  std::vector<T> data;
};

template<typename T>
class COOMatrix {
public:
  COOMatrix(dim_t rows, dim_t cols);
  void reset(const std::vector<dim_t> &row,
             const std::vector<dim_t> &col,
             const std::vector<T> &data);
  void reset(const std::vector<dim_t> &&row,
             const std::vector<dim_t> &&col,
             const std::vector<T> &&data);
  COOChunk<T>& data();
  const COOChunk<T>& data() const;
private:
  std::shared_ptr<COOChunk<T> > data_;
  dim_t rows_, cols_;
};

template<typename T>
COOMatrix<T>::COOMatrix(dim_t rows, dim_t cols) {
  CHECK_GE(rows, 0);
  CHECK_GE(cols, 0);
  rows_ = rows;
  cols_ = cols;
  data_.reset(new COOChunk<T>);
}

template<typename T>
void COOMatrix<T>::reset(const std::vector<dim_t> &row,
                         const std::vector<dim_t> &col,
                         const std::vector<T> &data) {
  COOChunk<T> &chunk = data();
  chunk.row = row;
  chunk.col = col;
  chunk.data = data;
}

template<typename T>
void COOMatrix<T>::reset(const std::vector<dim_t> &&row,
                         const std::vector<dim_t> &&col,
                         const std::vector<T> &&data) {
  COOChunk<T> &chunk = data();
  chunk.row = std::move(row);
  chunk.col = std::move(col);
  chunk.data = std::move(data);
}

template<typename T>
COOChunk<T>& COOMatrix<T>::data() {
  return *data_;
}

template<typename T>
const COOChunk<T>& COOMatrix<T>::data() const {
  return *data_;
}

#endif
