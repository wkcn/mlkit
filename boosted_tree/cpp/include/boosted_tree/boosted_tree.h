#ifndef BOOSTED_TREE_BOOSTED_TREE_H_
#define BOOSTED_TREE_BOOSTED_TREE_H_

#include <cmath>
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
  float gamma = 0;
  int n_jobs = 1;
  int seed = 39;
  std::string tree_method = "auto";  // ["auto", "exact", "approx"]
  float sketch_eps = 0.03;
};
/*
 * the samples will be groups per TREE_METHOD_APPROX_RATIO / sketch_eps samples,
 * and approximate into 1.0 / sketch_eps samples
 */
const int TREE_METHOD_APPROX_RATIO = 100;

class BoostedTree {
 public:
  BoostedTree(const BoostedTreeParam &);
  virtual ~BoostedTree();
  void train(const CSRMatrix<float> &X, const Vec<float> &Y);
  Vec<float> predict(const CSRMatrix<float> &X);

 public:
  static constexpr float MISSING_VALUE = nanf("");

 private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};

#endif
