#pragma once

#include <numeric>
#include <mutex>
#include <queue>
#include <vector>

#include <boosted_tree/csr_matrix.h>
#include <boosted_tree/vec.h>

struct Node {
  int left, right;
};

class BoostedTree::Impl {
public:
  void train(const CSRMatrix<float> &X, const Vec<float> &Y);
  Vec<float> predict(const CSRMatrix<float> &X);
private:
  int GetNewNodeID();
  int CreateNode(const std::vector<int> &sample_ids, const std::vector<int> &feature_ids);
private:
  int root;
  std::vector<Node> nodes_;
  std::queue<int> free_nodes_queue_;
  std::mutex nodes_alloc_mtx_;
  CSRMatrix<float> XT_;
  Vec<float> Y_;
};
