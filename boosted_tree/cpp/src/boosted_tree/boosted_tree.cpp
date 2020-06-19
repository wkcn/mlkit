#include <iostream>
#include <vector>
#include <numeric>

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
  Vec<float> residual(Y_);
  int iter = 0;
  while (1) {
    LOG(INFO) << "Iteration: " << iter;
    int root = CreateNode(residual, feature_ids, feature_ids);
    trees.push_back(root);
    // returned residual is the predicted value
    // TODO: Check accuracy
    ++iter;
  }
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

int BoostedTree::Impl::CreateNode(Vec<float> &residual, const std::vector<int> &sample_ids, const std::vector<int> &feature_ids) {
  const size_t num_samples = sample_ids.size();
  Vec<float> part_residual(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    part_residual[i] = residual[sample_ids[i]];
  }
  float pred = loss.predict(part_residual);
  Vec<float> gradients(num_samples), hessians(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    gradients[i] = loss.gradient(pred, part_residual[i]);
  }
  for (int i = 0; i < num_samples; ++i) {
    hessians[i] = loss.hessian(pred, part_residual[i]);
  }
  // compute gradient and hessian
  for (int feature_id : feature_ids) {
    SplitInfo split_info = GetSplitInfo(sample_ids, feature_id, gradients, hessians);
  }
  return 0;
}

SplitInfo BoostedTree::Impl::GetSplitInfo(const std::vector<int> &sample_ids, int feature_id, const Vec<float> &gradients, const Vec<float> &hessians) {
  CSRRow sfeat = XT_[feature_id];
  const size_t num_samples = sample_ids.size();
  Vec<float> feat = sfeat.at(sample_ids.begin(), sample_ids.end());
  std::vector<int> inds(num_samples);
  std::iota(inds.begin(), inds.end(), 0);
  std::sort(inds.begin(), inds.end(), [&feat](const int a, const int b) {
      return feat[a] < feat[b];
  });
  size_t num_split = 0;
  for (int i = 1; i < inds.size(); ++i) {
    if (feat[inds[i - 1]] != feat[inds[i]]) {
      ++num_split;
    }
  }
  Vec<float> split(num_split);
  float last = feat[inds[0]];
  for (int i = 1, j = 0; i < inds.size(); ++i) {
    if (feat[inds[i - 1]] != feat[inds[i]]) {
      float v = feat[inds[i]];
      // get split
      split[j++] = (last + v) / 2.0;
      last = v;
    }
  }
}
