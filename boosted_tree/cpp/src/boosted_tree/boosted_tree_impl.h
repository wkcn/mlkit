#pragma once

#include <boosted_tree/array.h>
#include <boosted_tree/boosted_tree.h>
#include <boosted_tree/csr_matrix.h>
#include <boosted_tree/objective.h>
#include <boosted_tree/vec.h>

#include <array>
#include <string>
#include <sstream>
#include <mutex>
#include <numeric>
#include <queue>
#include <vector>

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

struct GradientInfo {
  float gradient, hessian;
  GradientInfo() : gradient(0), hessian(0){};
  GradientInfo(float value) : gradient(value), hessian(value){};
  GradientInfo(float g, float h) : gradient(g), hessian(h){};
  GradientInfo &operator+=(const GradientInfo &b) {
    gradient += b.gradient;
    hessian += b.hessian;
    return *this;
  }
  GradientInfo operator+(const GradientInfo &b) const {
    GradientInfo info(gradient + b.gradient, hessian + b.hessian);
    return info;
  }
  bool operator<(const GradientInfo &b) const { return hessian < b.hessian; }
  operator float() const { return hessian; }
};

class BoostedTree::Impl {
 public:
  Impl(const BoostedTreeParam &);
  void train(const CSRMatrix<float> &X, const Vec<float> &Y);
  Vec<float> predict(const CSRMatrix<float> &X) const;
  float predict_one(const CSRRow<float> &X) const;
  std::string str() const;

 private:
  float predict_one_in_a_tree(const CSRRow<float> &X, int root) const;
  int GetNewNodeID();
  int CreateNode(Vec<float> &integrals, const std::vector<int> &sample_ids,
                 const std::vector<int> &feature_ids, const int depth);
  inline float GetGain(float G, float H) const;
  SplitInfo GetExactSplitInfo(const std::vector<int> &sample_ids,
                              int feature_id, const Vec<float> &gradients,
                              const float G_sum, const Vec<float> &hessians,
                              const float H_sum);

  SplitInfo GetApproxSplitInfo(const std::vector<int> &sample_ids,
                               int feature_id, const Vec<float> &gradients,
                               const float G_sum, const Vec<float> &hessians,
                               const float H_sum);

 private:
  template <typename DType, typename IType>
  Vec<DType> ReorderVec(const Vec<DType> &data, const std::vector<IType> &inds);

 private:
  BoostedTreeParam param_;
  std::vector<int> trees;
  Objective<float> *objective;
  std::vector<Node *> nodes_;
  std::queue<int> free_nodes_queue_;
  std::mutex nodes_alloc_mtx_;
  CSRMatrix<float> XT_;
  Vec<float> Y_;
};
