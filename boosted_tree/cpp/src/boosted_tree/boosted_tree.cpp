#include <cfloat>
#include <iostream>
#include <vector>
#include <numeric>

#include <boosted_tree/boosted_tree.h>
#include <boosted_tree/logging.h>

#include "./boosted_tree_impl.h"

BoostedTree::BoostedTree(const BoostedTreeParam &param) : pImpl(new Impl(param)) {
}

BoostedTree::~BoostedTree() = default;

void BoostedTree::train(const CSRMatrix<float> &X, const Vec<float> &Y) {
  pImpl->train(X, Y);
}

Vec<float> BoostedTree::predict(const CSRMatrix<float> &X) {
  return pImpl->predict(X);
}

BoostedTree::Impl::Impl(const BoostedTreeParam &param) : param_(param) {
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
  for (int iter = 1; iter <= param_.n_estimators; ++iter) {
    int root = CreateNode(residual, sample_ids, feature_ids, 1);
    trees.push_back(root);
    // returned residual is the predicted value
    // TODO: Check accuracy
    float loss = Sum(residual * residual);
    LOG(INFO) << "Iteration: " << iter << " Loss: " << loss;
    if (loss <= 1e-3) break;
  }
  LOG(INFO) << "Train over";
}

Vec<float> BoostedTree::Impl::predict(const CSRMatrix<float> &X) {
  const int N = X.length();
  Vec<float> preds(N);
  for (int i = 0; i < N; ++i) {
    CSRRow x = X[i];
    preds[i] = predict_one(x); 
  }
  return preds;
}

float BoostedTree::Impl::predict_one(const CSRRow<float> &X) {
  float out = 0;
  for (int root : trees) {
    out += predict_one_in_a_tree(X, root);
  }
  return out;
}

float BoostedTree::Impl::predict_one_in_a_tree(const CSRRow<float> &X, int root) {
  while (1) {
    const Node &node = *nodes_[root];
    if (node.is_leaf) return node.value;
    bool is_left = X[node.feature_id] < node.value;
    root = is_left ? node.left : node.right; 
  }
  return 0;
}

int BoostedTree::Impl::GetNewNodeID() {
  std::lock_guard<std::mutex> lck(nodes_alloc_mtx_);
  int id;
  if (free_nodes_queue_.empty()) {
    id = nodes_.size();
    nodes_.resize(id + 1);
    nodes_[id] = new Node();
  } else {
    id = free_nodes_queue_.front();
    free_nodes_queue_.pop();
  }
  TEST_LT(id, nodes_.size());
  return id;
}

int BoostedTree::Impl::CreateNode(Vec<float> &residual, const std::vector<int> &sample_ids, const std::vector<int> &feature_ids, const int depth) {

  const int nid = GetNewNodeID();
  Node &node = *nodes_[nid];

  const size_t num_samples = sample_ids.size();
  Vec<float> part_residual(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    part_residual[i] = residual[sample_ids[i]];
  }

  float pred = loss.predict(part_residual);

  bool gen_leaf = true;
  if (param_.max_depth == -1 || depth <= param_.max_depth) {
    if (sample_ids.size() > 1) {
      // TODO: 如何在回归问题中中止
      gen_leaf = false;
    }
  }
  if (!gen_leaf && !feature_ids.empty()) {
    // compute gradient and hessian
    Vec<float> gradients(num_samples), hessians(num_samples);
    for (int i = 0; i < num_samples; ++i) {
      gradients[i] = loss.gradient(pred, part_residual[i]);
    }
    for (int i = 0; i < num_samples; ++i) {
      hessians[i] = loss.hessian(pred, part_residual[i]);
    }
    const float G_sum = Sum(gradients);
    const float H_sum = Sum(hessians);

    float best_gain = FLT_MIN;
    SplitInfo best_info;
    std::vector<int> new_feature_ids;
    for (int feature_id : feature_ids) {
      SplitInfo info = GetSplitInfo(sample_ids, feature_id, gradients, G_sum, hessians, H_sum);
      if (info.feature_id == -1) continue;
      new_feature_ids.push_back(info.feature_id);
      if (info.gain > best_gain) {
        best_gain = info.gain;
        best_info = info;
      }
    }

    if (best_gain != FLT_MIN) {
      // split
      node.is_leaf = false;
      node.feature_id = best_info.feature_id;

      std::vector<int> left_sample_ids, right_sample_ids;
      float split = best_info.split;
      CSRRow sfeat = XT_[best_info.feature_id];
      Vec<float> feat = sfeat.at(sample_ids.begin(), sample_ids.end());
      for (int i = 0; i < num_samples; ++i) {
        if (feat[i] < split) left_sample_ids.push_back(sample_ids[i]);
        else right_sample_ids.push_back(sample_ids[i]);
      }

      // subtree
      node.left = CreateNode(residual, left_sample_ids, new_feature_ids, depth + 1);
      node.right = CreateNode(residual, right_sample_ids, new_feature_ids, depth + 1);
      return nid;
    }
  }

  // leaf
  node.is_leaf = true;
  param_.learning_rate = 1;
  float pred_factor = pred * param_.learning_rate;
  node.value = pred_factor;
  // update residual
  for (int i = 0; i < num_samples; ++i) {
    float &r = residual[sample_ids[i]];
    r -= pred_factor;
  }
  return nid; 
}

SplitInfo BoostedTree::Impl::GetSplitInfo(const std::vector<int> &sample_ids, int feature_id, const Vec<float> &gradients, const float G_sum, const Vec<float> &hessians, const float H_sum) {
  CSRRow sfeat = XT_[feature_id];
  const size_t num_samples = sample_ids.size();
  Vec<float> feat = sfeat.at(sample_ids.begin(), sample_ids.end());
  std::vector<int> inds(num_samples);
  std::iota(inds.begin(), inds.end(), 0);
  std::sort(inds.begin(), inds.end(), [&feat](const int a, const int b) {
      return feat[a] < feat[b];
  });
  size_t num_splits = 0;
  for (int i = 1; i < inds.size(); ++i) {
    if (feat[inds[i - 1]] != feat[inds[i]]) {
      ++num_splits;
    }
  }
  if (num_splits == 0) {
    SplitInfo info;
    info.feature_id = -1;
    return info;
  }
  TEST_GT(num_splits, 0);
  Vec<float> splits(num_splits);
  float last = feat[inds[0]];
  for (int i = 1, j = 0; i < inds.size(); ++i) {
    if (feat[inds[i - 1]] != feat[inds[i]]) {
      float v = feat[inds[i]];
      // get split
      splits[j++] = (last + v) / 2.0;
      last = v;
    }
  }
  float G_L = 0, H_L = 0;
  int si = 0;
  float best_gain = FLT_MIN;
  float best_split;
  for (float split : splits) {
    while (si < num_samples && feat[inds[si]] < split) {
      int ind = inds[si];
      G_L += gradients[ind]; 
      H_L += hessians[ind]; 
      ++si;
    }
    float G_R = G_sum - G_L;
    float H_R = H_sum - H_L;
    float gain = G_L * G_L / (H_L + param_.reg_lambda) + G_R * G_R / (H_R + param_.reg_lambda);
    if (gain > best_gain) {
      best_gain = gain;
      best_split = split;
    }
  }
  SplitInfo info;
  info.feature_id = feature_id;
  info.split = best_split;
  info.gain = best_gain;
  return info;
}
