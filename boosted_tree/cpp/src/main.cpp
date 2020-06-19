#include <vector>
#include <string>

#include <boosted_tree/boosted_tree.h>
#include <boosted_tree/csr_matrix.h>
#include <boosted_tree/io.h>
#include <boosted_tree/logging.h>

float ComputeAccuracy(const Vec<float> &a, const Vec<float> &b) {
  if (a.size() != b.size()) return 0;
  int right = 0;
  for (int i = 0; i < a.size(); ++i) {
    if ((a[i] >= 0.5) == (b[i] >= 0.5)) ++right;
  }
  return float(right) / a.size();
}

int main(int argc, char **argv) {
  BoostedTree bst;
  CSRMatrix<float> X(0, 0);
  Vec<float> Y;
  if (argc > 1) {
    std::string filename = argv[1];
    LOG(INFO) << "Open data: " << filename;
    auto p = ReadLibSVMFile<float, float>(filename);
    X = std::move(p.first);
    Y = std::move(p.second);
    bst.train(X, Y);
    float acc = ComputeAccuracy(bst.predict(X), Y);
    LOG(INFO) << "Training Accuracy: " << acc;
  } else {
    LOG(INFO) << "Please pass the argument `filename`";
  }
  return 0;
}
