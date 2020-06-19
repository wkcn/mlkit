#pragma once

#include <vector>

#include <boosted_tree/csr_matrix.h>

struct Node {
  int left, right;
};

class BoostedTree::Impl {
public:
  void train(const CSRMatrix<float> &X, const Vec<float> &Y);
  Vec<float> predict(const CSRMatrix<float> &X);
private:
  std::vector<Node> nodes;
};
