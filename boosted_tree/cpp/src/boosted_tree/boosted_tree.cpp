#include <boosted_tree/boosted_tree.h>
#include <boosted_tree/logging.h>
#include <boosted_tree/quantile.h>
#include <omp.h>

#include <cfloat>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <set>
#include <stack>
#include <utility>
#include <vector>

#include "./boosted_tree_impl.h"

BoostedTree::BoostedTree(const BoostedTreeParam &param)
    : pImpl(new Impl(param)) {}

BoostedTree::~BoostedTree() = default;

void BoostedTree::train(const CSRMatrix<float> &X, const Vec<float> &Y) {
  pImpl->train(X, Y);
}

Vec<float> BoostedTree::predict(const CSRMatrix<float> &X) const {
  return pImpl->predict(X);
}

std::string BoostedTree::str() const { return pImpl->str(); }

BoostedTree::Impl::Impl(const BoostedTreeParam &param) : param_(param) {
  objective = Registry<Objective<float>>::Find(param_.objective);
  if (objective == nullptr) {
    std::string msg =
        "objective function [" + param_.objective + "] not found\n";
    msg += "Supported objective functions: ";
    std::vector<std::string> names = Registry<Objective<float>>::List();
    bool first = true;
    for (const std::string &name : names) {
      if (first)
        first = false;
      else
        msg += ", ";
      msg += name;
    }
    LOG(FATAL) << msg;
  }
  std::set<std::string> tree_methods{"auto", "exact", "approx"};
  CHECK(tree_methods.count(param_.tree_method))
      << "Not supported " << param_.tree_method
      << ", tree_method should be in [\"auto\", \"exact\", \"approx\"]";
  CHECK(param_.subsample >= 0 && param_.subsample <= 1)
      << "subsample should be in [0, 1]";
}

void BoostedTree::Impl::train(const CSRMatrix<float> &X, const Vec<float> &Y) {
  srand(param_.seed);
  const int num_samples = X.length();
  const int num_features = X[0].length();
  CHECK_EQ(num_samples, Y.size());
  LOG(INFO) << "Input Data: (" << num_samples << " X " << num_features << ")";
  XT_ = X.transpose();
  Y_ = std::move(Y);
  LOG(INFO) << "Start training...";
  std::vector<int> sample_ids(num_samples);
  std::iota(sample_ids.begin(), sample_ids.end(), 0);
  std::vector<int> feature_ids(num_features);
  std::iota(feature_ids.begin(), feature_ids.end(), 0);
  Vec<float> integrals(num_samples);
  integrals = 0;
  for (int iter = 1; iter <= param_.n_estimators; ++iter) {
    int root = CreateNode(integrals, sample_ids, feature_ids, 1);
    trees.push_back(root);
    Vec<float> pred = predict(X);
    float loss = 0;
    for (int i = 0; i < num_samples; ++i) {
      loss += objective->compute(pred[i], Y_[i]);
    }
    loss /= num_samples;
    LOG(INFO) << "Iteration: " << iter << " Loss: " << loss;
    if (loss <= 1e-3) break;
  }
}

Vec<float> BoostedTree::Impl::predict(const CSRMatrix<float> &X) const {
  const int N = X.length();
  Vec<float> preds(N);
#pragma omp parallel for num_threads(param_.n_jobs)
  for (int i = 0; i < N; ++i) {
    CSRRow<float> x = X[i];
    preds[i] = predict_one(x);
  }
  return preds;
}

float BoostedTree::Impl::predict_one(const CSRRow<float> &X) const {
  float out = 0;
  for (int root : trees) {
    out += predict_one_in_a_tree(X, root);
  }
  return objective->predict(out);
}

float BoostedTree::Impl::predict_one_in_a_tree(const CSRRow<float> &X,
                                               int root) const {
  while (1) {
    const Node &node = *nodes_[root];
    if (node.is_leaf) return node.value;
    const float feat = X[node.feature_id];
    bool is_left = std::isnan(feat) ? node.miss_left : feat < node.value;
    root = is_left ? node.left : node.right;
  }
  return 0;
}

std::string BoostedTree::Impl::str() const {
  const size_t num_trees = trees.size();
  std::stringstream ss;
  for (int t = 0; t < num_trees; ++t) {
    ss << "Tree " << t + 1 << ":\n";
    std::function<void(const int, const int)> F;
    F = [&](const int nid, const int height) {
      const Node &node = *nodes_[nid];
      std::string space(height, '\t');
      if (node.is_leaf) {
        ss << space << "predict: " << node.value << '\n';
      } else {
        ss << space << "f" << node.feature_id << " < " << node.value;
        if (node.miss_left) ss << " or missing";
        ss << '\n';
        F(node.left, height + 1);
        ss << space << "f" << node.feature_id << " >= " << node.value;
        if (!node.miss_left) ss << " or missing";
        ss << '\n';
        F(node.right, height + 1);
      }
    };
    F(trees[t], 1);
  }
  return ss.str();
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
  DCHECK_LT(id, nodes_.size());
  return id;
}

int BoostedTree::Impl::CreateNode(Vec<float> &integrals,
                                  const std::vector<int> &sample_ids,
                                  const std::vector<int> &feature_ids,
                                  const int depth) {
  const int nid = GetNewNodeID();
  Node &node = *nodes_[nid];

  const size_t num_samples = sample_ids.size();
  const size_t num_subsamples =
      std::max(static_cast<size_t>(1),
               static_cast<size_t>(num_samples * param_.subsample));
  std::vector<int> subsample_ids = sample_ids;
  if (param_.subsample < 1) {
    std::random_shuffle(subsample_ids.begin(), subsample_ids.end());
    subsample_ids.resize(num_subsamples);
  }

  Vec<float> part_integrals(num_subsamples);
  for (int i = 0; i < num_subsamples; ++i) {
    part_integrals[i] = integrals[subsample_ids[i]];
  }
  Vec<float> part_labels(num_subsamples);
  for (int i = 0; i < num_subsamples; ++i) {
    part_labels[i] = Y_[subsample_ids[i]];
  }

  bool gen_leaf = true;
  if (param_.max_depth <= 0 || depth <= param_.max_depth) {
    if (num_subsamples > 1) {
      // TODO: 如何在回归问题中中止
      gen_leaf = false;
    }
  }
  // compute gradient and hessian
  Vec<float> gradients(num_subsamples), hessians(num_subsamples);
  for (int i = 0; i < num_subsamples; ++i) {
    gradients[i] = objective->gradient(part_integrals[i], part_labels[i]);
  }
  for (int i = 0; i < num_subsamples; ++i) {
    hessians[i] = objective->hessian(part_integrals[i], part_labels[i]);
  }
  const float G_sum = Sum(gradients);
  const float H_sum = Sum(hessians);
  float best_gain = GetGain(G_sum, H_sum) + param_.gamma * 2;

  bool using_exact_hist =
      (param_.tree_method == "exact" ||
       num_subsamples <= size_t(TREE_METHOD_APPROX_RATIO / param_.sketch_eps));
  if (!gen_leaf && !feature_ids.empty()) {
    SplitInfo best_info;
    best_info.feature_id = -1;
    std::vector<int> new_feature_ids;
    const int num_features = feature_ids.size();
#pragma omp parallel for num_threads(param_.n_jobs)
    for (int i = 0; i < num_features; ++i) {
      int feature_id = feature_ids[i];
      SplitInfo info =
          using_exact_hist
              ? GetExactSplitInfo(subsample_ids, feature_id, gradients, G_sum,
                                  hessians, H_sum)
              : GetApproxSplitInfo(subsample_ids, feature_id, gradients, G_sum,
                                   hessians, H_sum);
#pragma omp critical
      if (info.feature_id != -1) {
        new_feature_ids.push_back(info.feature_id);
        if (info.gain > best_gain) {
          best_gain = info.gain;
          best_info = info;
        }
      }
    }

    if (best_info.feature_id != -1) {
      // split
      node.is_leaf = false;
      node.feature_id = best_info.feature_id;
      node.miss_left = best_info.miss_left;
      const float split = best_info.split;
      node.value = split;

      std::vector<int> left_sample_ids, right_sample_ids;
      CSRRow<float> sfeat = XT_[best_info.feature_id];
      Vec<float> feat = sfeat.at(sample_ids.begin(), sample_ids.end());
      for (int i = 0; i < num_samples; ++i) {
        /*
         * left:
         *   isnan(feat[i]) == false && feat[i] < split
         *   isnan(feat[i]) == true && node.miss_left == true
         *   (A && B) || (!A && C)
         */
        if ((!std::isnan(feat[i]) && feat[i] < split) ||
            (std::isnan(feat[i]) && node.miss_left))
          left_sample_ids.push_back(sample_ids[i]);
        else
          right_sample_ids.push_back(sample_ids[i]);
      }

      // subtree
      node.left =
          CreateNode(integrals, left_sample_ids, new_feature_ids, depth + 1);
      node.right =
          CreateNode(integrals, right_sample_ids, new_feature_ids, depth + 1);
      return nid;
    }
  }

  // leaf
  node.is_leaf = true;
  // float mean_integral = Mean(part_integrals);
  // float pred = objective->estimate(part_labels) - mean_integral;
  float pred = -G_sum / (H_sum + param_.reg_lambda);
  float pred_factor = pred * param_.learning_rate;
  node.value = pred_factor;
  // update integrals
  for (int i = 0; i < num_samples; ++i) {
    float &r = integrals[sample_ids[i]];
    r += pred_factor;
  }
  return nid;
}

float BoostedTree::Impl::GetGain(float G, float H) const {
  float gain = G * G / (H + param_.reg_lambda);
  return gain;
}

template <typename DType, typename IType>
Vec<DType> BoostedTree::Impl::ReorderVec(const Vec<DType> &data,
                                         const std::vector<IType> &inds) {
  const size_t size = inds.size();
  if (size == 0) return {};
  Vec<DType> out(size);
  for (int i = 0; i < (int)size; ++i) {
    out[i] = data[inds[i]];
  }
  return out;
}

SplitInfo BoostedTree::Impl::GetExactSplitInfo(
    const std::vector<int> &sample_ids, int feature_id,
    const Vec<float> &gradients, const float G_sum, const Vec<float> &hessians,
    const float H_sum) {
  // Basic exact greedy algorithm
  CSRRow<float> sfeat = XT_[feature_id];
  const size_t num_samples = sample_ids.size();
  Vec<float> feat = sfeat.at(sample_ids.begin(), sample_ids.end());
  std::vector<int> inds(num_samples);
  float G_missing = 0, H_missing = 0;
  int j = 0;
  // remove nan in inds
  for (int i = 0; i < num_samples; ++i) {
    if (std::isnan(feat[i])) {
      G_missing += gradients[i];
      H_missing += hessians[i];
    } else {
      inds[j++] = i;
    }
  }
  inds.resize(j);
  std::sort(inds.begin(), inds.end(),
            [&feat](const int a, const int b) { return feat[a] < feat[b]; });
  const size_t num_nonmiss_samples = j;
  const bool exist_missing = num_nonmiss_samples < num_samples;
  size_t num_splits = 0;
  for (int i = 1; i < num_nonmiss_samples; ++i) {
    if (feat[inds[i - 1]] != feat[inds[i]]) {
      ++num_splits;
    }
  }
  if (num_splits == 0) {
    SplitInfo info;
    info.feature_id = -1;
    return info;
  }
  DCHECK_GT(num_splits, 0);
  Vec<float> splits(num_splits);
  float last = feat[inds[0]];
  for (int i = 1, j = 0; i < num_nonmiss_samples; ++i) {
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
  bool best_miss_left;

  Vec<float> feat_cache = ReorderVec(feat, inds);
  Vec<float> gradients_cache = ReorderVec(gradients, inds);
  Vec<float> hessians_cache = ReorderVec(hessians, inds);

  for (float split : splits) {
    while (si < num_nonmiss_samples && feat_cache[si] < split) {
      G_L += gradients_cache[si];
      H_L += hessians_cache[si];
      ++si;
    }
    {
      // try enumerate missing value goto right
      float G_R = G_sum - G_L;
      float H_R = H_sum - H_L;
      float gain = GetGain(G_L, H_L) + GetGain(G_R, H_R);
      if (gain > best_gain) {
        best_gain = gain;
        best_split = split;
        best_miss_left = false;
      }
    }
    if (exist_missing) {
      // try enumerate missing value goto left
      float G_L2 = G_L + G_missing;
      float H_L2 = H_L + H_missing;
      float G_R2 = G_sum - G_L2;
      float H_R2 = H_sum - H_L2;
      float gain = GetGain(G_L2, H_L2) + GetGain(G_R2, H_R2);
      if (gain > best_gain) {
        best_gain = gain;
        best_split = split;
        best_miss_left = true;
      }
    }
  }
  SplitInfo info;
  info.feature_id = feature_id;
  info.split = best_split;
  info.gain = best_gain;
  info.miss_left = best_miss_left;
  return info;
}

SplitInfo BoostedTree::Impl::GetApproxSplitInfo(
    const std::vector<int> &sample_ids, int feature_id,
    const Vec<float> &gradients, const float G_sum, const Vec<float> &hessians,
    const float H_sum) {
  // Weighted quantile sketch
  CSRRow<float> sfeat = XT_[feature_id];
  const size_t num_samples = sample_ids.size();
  Vec<float> feat = sfeat.at(sample_ids.begin(), sample_ids.end());
  std::vector<int> inds(num_samples);
  float G_missing = 0, H_missing = 0;
  int j = 0;
  // remove nan in inds
  for (int i = 0; i < num_samples; ++i) {
    if (std::isnan(feat[i])) {
      G_missing += gradients[i];
      H_missing += hessians[i];
    } else {
      inds[j++] = i;
    }
  }
  inds.resize(j);
  using pair_t = std::pair<float, GradientInfo>;
  using quantile_t = Quantile<float, GradientInfo>;
  using summary_t = quantile_t::Summary;
  const size_t buffer_size = TREE_METHOD_APPROX_RATIO / param_.sketch_eps;
  const size_t num_buckets = 1.0 / param_.sketch_eps;
  std::vector<pair_t> buf(buffer_size);
  int buf_i = 0;
  summary_t summary;
  for (int ind : inds) {
    buf[buf_i++] =
        pair_t{feat[ind], GradientInfo(gradients[ind], hessians[ind])};
    if (buf_i >= buffer_size) {
      // merge then prune
      summary_t tmp_summary(buf);
      tmp_summary = quantile_t::Prune(tmp_summary, num_buckets);
      summary = quantile_t::Merge(summary, tmp_summary);
      buf_i = 0;
    }
  }
  if (buf_i > 0) {
    buf.resize(buf_i);
    summary = quantile_t::Merge(summary, summary_t(buf));
  }
  const bool exist_missing = inds.size() < num_samples;
  size_t num_splits = summary.size();
  if (num_splits == 0) {
    SplitInfo info;
    info.feature_id = -1;
    return info;
  }
  DCHECK_GT(num_splits, 0);
  float best_gain = FLT_MIN;
  float best_split;
  bool best_miss_left;
  for (int i = 0; i < summary.size(); ++i) {
    const auto &entry = summary[i];
    // update split, G_L and H_L
    float split = entry.value;
    const GradientInfo &ginfo = entry.rmin;
    float G_L = ginfo.gradient;
    float H_L = ginfo.hessian;
    {
      // try enumerate missing value goto right
      float G_R = G_sum - G_L;
      float H_R = H_sum - H_L;
      float gain = GetGain(G_L, H_L) + GetGain(G_R, H_R);
      if (gain > best_gain) {
        best_gain = gain;
        best_split = split;
        best_miss_left = false;
      }
    }
    if (exist_missing) {
      // try enumerate missing value goto left
      float G_L2 = G_L + G_missing;
      float H_L2 = H_L + H_missing;
      float G_R2 = G_sum - G_L2;
      float H_R2 = H_sum - H_L2;
      float gain = GetGain(G_L2, H_L2) + GetGain(G_R2, H_R2);
      if (gain > best_gain) {
        best_gain = gain;
        best_split = split;
        best_miss_left = true;
      }
    }
  }
  SplitInfo info;
  info.feature_id = feature_id;
  info.split = best_split;
  info.gain = best_gain;
  info.miss_left = best_miss_left;
  return info;
}
