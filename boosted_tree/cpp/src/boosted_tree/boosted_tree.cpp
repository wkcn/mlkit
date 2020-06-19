#include <iostream>

#include <boosted_tree/boosted_tree.h>
#include <boosted_tree/logging.h>

#include "./boosted_tree_impl.h"

BoostedTree::BoostedTree() : pImpl(new Impl) {
}

BoostedTree::~BoostedTree() = default;

void BoostedTree::train(const CSRMatrix<float> &X, const Vec<float> &Y) {
  pImpl->train(X, Y);
}

Vec<float> BoostedTree::predict(const CSRMatrix<float> &X) {
  return pImpl->predict(X);
}


void BoostedTree::Impl::train(const CSRMatrix<float> &X, const Vec<float> &Y) {
  const int N = X.length();
  const int M = X[0].length();
  CHECK_EQ(N, Y.size());
  LOG(INFO) << "Input Data: (" << N << " X " << M << ")";
  CSRMatrix<float> XT = X.transpose();
  LOG(INFO) << "Start training...";
}

Vec<float> BoostedTree::Impl::predict(const CSRMatrix<float> &X) {
  return {};
}
