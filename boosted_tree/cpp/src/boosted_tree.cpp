#include <boosted_tree/boosted_tree.h>

void BoostedTree::train(const CSRMatrix<float> &X, const std::vector<float> &Y) {
  CSRMatrix<float> XT = X.transpose();
}
