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
  XT_ = X.transpose();
  Y_ = std::move(Y);
  LOG(INFO) << "Start training...";
  std::vector<int> sample_ids(N);
  std::iota(sample_ids.begin(), sample_ids.end(), 0);
  std::vector<int> feature_ids(M);
  std::iota(feature_ids.begin(), feature_ids.end(), 0);
  root = CreateNode(feature_ids, feature_ids);
}

Vec<float> BoostedTree::Impl::predict(const CSRMatrix<float> &X) {
  return {};
}

int BoostedTree::Impl::GetNewNodeID() {
  std::lock_guard<std::mutex> lck(nodes_alloc_mtx_);
  int id;
  if (free_nodes_queue_.empty()) {
    id = nodes_.size();
    nodes_.resize(id + 1);
  } else {
    id = free_nodes_queue_.front();
    free_nodes_queue_.pop();
  }
  return id;
}

int BoostedTree::Impl::CreateNode(const std::vector<int> &sample_ids, const std::vector<int> &feature_ids) {
  // compute gradient and hessian
  for (int feature_id : feature_ids) {
  }
  return 0;
}
