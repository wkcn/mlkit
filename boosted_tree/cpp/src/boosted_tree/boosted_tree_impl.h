#pragma once

#include <array>
#include <numeric>
#include <mutex>
#include <queue>
#include <vector>

#include <boosted_tree/csr_matrix.h>
#include <boosted_tree/loss.h>
#include <boosted_tree/vec.h>
#include <boosted_tree/array.h>

struct Node {
  /*
   * Inner Node
   *   is_leaf = false
   *   feature_id: feature_id
   *   left: [, value)
   *   right: [value, )
   *
   * Leaf
   *   is_leaf = true
   *   predict: value
   */
  int left, right;
  int feature_id;
  float value;
  bool is_leaf;
};

struct SplitInfo {
  int feature_id;
  float split;
  float gain;
};

class BoostedTree::Impl {
public:
  Impl();
  void train(const CSRMatrix<float> &X, const Vec<float> &Y);
  Vec<float> predict(const CSRMatrix<float> &X);
  float predict_one(const CSRRow<float> &X);
private:
  float predict_one_in_a_tree(const CSRRow<float> &X, int root);
  int GetNewNodeID();
  int CreateNode(Vec<float> &residual, const std::vector<int> &sample_ids, const std::vector<int> &feature_ids);
  SplitInfo GetSplitInfo(const std::vector<int> &sample_ids, int feature_id, const Vec<float> &gradients, const float G_sum, const Vec<float> &hessians, const float H_sum);
private:
  std::vector<int> trees;
  SquareLoss loss;
  std::vector<Node*> nodes_;
  std::queue<int> free_nodes_queue_;
  std::mutex nodes_alloc_mtx_;
  CSRMatrix<float> XT_;
  Vec<float> Y_;
  float lambda;
};
