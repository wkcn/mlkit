#ifndef BOOSTED_TREE_BOOSTED_TREE_H_
#define BOOSTED_TREE_BOOSTED_TREE_H_

#include <memory>
#include <string>
#include <vector>

#include "./csr_matrix.h"

struct BoostedTreeParam {
  int max_depth = 6;
  float learning_rate = 0.3;  // eta
  int n_estimators = 100;
  std::string objective = "reg:linear";
  float reg_lambda = 1;
};

class BoostedTree {
public:
  BoostedTree(const BoostedTreeParam &);
  virtual ~BoostedTree();
  void train(const CSRMatrix<float> &X, const Vec<float> &Y);
  Vec<float> predict(const CSRMatrix<float> &X);
private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};

#endif
