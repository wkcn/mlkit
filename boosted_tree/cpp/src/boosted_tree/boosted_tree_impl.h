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
};

class BoostedTree::Impl {
public:
  void train(const CSRMatrix<float> &X, const Vec<float> &Y);
  Vec<float> predict(const CSRMatrix<float> &X);
private:
  int GetNewNodeID();
  int CreateNode(Vec<float> &residual, const std::vector<int> &sample_ids, const std::vector<int> &feature_ids);
  SplitInfo GetSplitInfo(const std::vector<int> &sample_ids, int feature_id, const Vec<float> &gradients, const Vec<float> &hessians);
private:
  std::vector<int> trees;
  LogisticLoss loss;
  std::vector<Node> nodes_;
  std::queue<int> free_nodes_queue_;
  std::mutex nodes_alloc_mtx_;
  CSRMatrix<float> XT_;
  Vec<float> Y_;
};
