#pragma once

#include <array>
#include <numeric>
#include <mutex>
#include <queue>
#include <vector>

#include <boosted_tree/boosted_tree.h>
#include <boosted_tree/array.h>
#include <boosted_tree/vec.h>
#include <boosted_tree/csr_matrix.h>
#include <boosted_tree/objective.h>

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
  bool miss_left;
};

struct SplitInfo {
  int feature_id;
  float split;
  float gain;
  bool miss_left;
};

class BoostedTree::Impl {
public:
  Impl(const BoostedTreeParam&);
  void train(const CSRMatrix<float> &X, const Vec<float> &Y);
  Vec<float> predict(const CSRMatrix<float> &X);
  float predict_one(const CSRRow<float> &X);
private:
  float predict_one_in_a_tree(const CSRRow<float> &X, int root);
  int GetNewNodeID();
  int CreateNode(Vec<float> &integrals, const std::vector<int> &sample_ids, const std::vector<int> &feature_ids, const int depth);
  inline float GetGain(float G, float H) const;
  SplitInfo GetSplitInfo(const std::vector<int> &sample_ids, int feature_id, const Vec<float> &gradients, const float G_sum, const Vec<float> &hessians, const float H_sum);
private:
  BoostedTreeParam param_;
  std::vector<int> trees;
  ObjectiveBase *objective;
  std::vector<Node*> nodes_;
  std::queue<int> free_nodes_queue_;
  std::mutex nodes_alloc_mtx_;
  CSRMatrix<float> XT_;
  Vec<float> Y_;
};
