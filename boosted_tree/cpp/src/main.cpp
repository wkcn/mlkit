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
  BoostedTreeParam param;
  BoostedTree bst(param);
  CSRMatrix<float> X(0, 0);
  Vec<float> Y;
  if (argc > 1) {
    std::string train_fname = argv[1];
    LOG(INFO) << "Open training data: " << train_fname;
    auto p = ReadLibSVMFile<float, float>(train_fname);
    X = std::move(p.first);
    Y = std::move(p.second);
    bst.train(X, Y);
    float acc = ComputeAccuracy(bst.predict(X), Y);
    LOG(INFO) << "Training Accuracy: " << acc;

    if (argc > 2) {
      std::string test_fname = argv[2];
      LOG(INFO) << "Open testing data: " << test_fname;
      auto p = ReadLibSVMFile<float, float>(test_fname);
      float acc = ComputeAccuracy(bst.predict(p.first), p.second);
      LOG(INFO) << "Testing Accuracy: " << acc;
    }
  } else {
    LOG(INFO) << "./main <train_fname> <test_fname>";
  }
  return 0;
}
