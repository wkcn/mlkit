#include <boosted_tree/boosted_tree.h>
#include "./boosted_tree_impl.h"

BoostedTree::BoostedTree() : pImpl(new Impl) {
}

BoostedTree::~BoostedTree() = default;

void BoostedTree::train(const CSRMatrix<float> &X, const std::vector<float> &Y) {
  pImpl->train(X, Y);
}

std::vector<float> BoostedTree::predict(const CSRMatrix<float> &X) {
  return pImpl->predict(X);
}


void BoostedTree::Impl::train(const CSRMatrix<float> &X, const std::vector<float> &Y) {
  CSRMatrix<float> XT = X.transpose();
  LOG(INFO) << "TRAIN";
}

std::vector<float> BoostedTree::Impl::predict(const CSRMatrix<float> &X) {
  return {};
}
