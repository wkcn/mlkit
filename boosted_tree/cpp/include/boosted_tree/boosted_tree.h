#ifndef BOOSTED_TREE_BOOSTED_TREE_H_
#define BOOSTED_TREE_BOOSTED_TREE_H_

#include <memory>
#include <vector>

#include "./csr_matrix.h"

class BoostedTree {
public:
  BoostedTree();
  virtual ~BoostedTree();
  void train(const CSRMatrix<float> &X, const Vec<float> &Y);
  Vec<float> predict(const CSRMatrix<float> &X);
private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};

#endif
