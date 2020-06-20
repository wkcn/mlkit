#include <cmath>
#include <cstdlib>
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

float ComputeRMSE(const Vec<float> &a, const Vec<float> &b) {
  if (a.size() != b.size()) return 0;
  Vec<float> tmp = a - b;
  std::for_each(tmp.begin(), tmp.end(), [](float &x){x *= x;});
  return sqrt(Mean(tmp));
}

void Evaluate(const Vec<float> &a, const Vec<float> &b, const std::string &prefix) {
  LOG(INFO) << prefix << " Accuracy: " << ComputeAccuracy(a, b);
  LOG(INFO) << prefix << " RMSE: " << ComputeRMSE(a, b);
}

void GenMissingValue(CSRMatrix<float> &X, float ratio) {
  if (ratio == 0) return;
  CHECK_GE(ratio, 0);
  CHECK_LE(ratio, 1);
  const int rows = X.length();
  const int cols = X[0].length();
  int num_missing = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const float p = float(rand() % 10000) / 10000;
      if (p <= ratio) {
        X[r].set(c, BoostedTree::MISSING_VALUE);
        ++num_missing;
      }
    }
  }
  LOG(INFO) << "The number of generated missing value is " << num_missing;
}

int main(int argc, char **argv) {
  BoostedTreeParam param;
  param.learning_rate = 1;
  param.n_estimators = 5;
  const float missing_ratio = 0;

  BoostedTree bst(param);

  if (argc > 1) {
    std::string train_fname = argv[1];
    LOG(INFO) << "Open training data: " << train_fname;
    auto p = ReadLibSVMFile<float, float>(train_fname);
    CSRMatrix<float> X = std::move(p.first);
    GenMissingValue(X, missing_ratio);
    Vec<float> Y = std::move(p.second);
    bst.train(X, Y);
    Vec<float> preds = bst.predict(X);
    Evaluate(preds, Y, "Training");

    if (argc > 2) {
      std::string test_fname = argv[2];
      LOG(INFO) << "Open testing data: " << test_fname;
      auto p = ReadLibSVMFile<float, float>(test_fname);
      CSRMatrix<float> testX = std::move(p.first);
      Vec<float> testY = std::move(p.second);
      GenMissingValue(testX, missing_ratio);
      Vec<float> testPreds = bst.predict(testX);
      Evaluate(testPreds, testY, "Testing");
    }
  } else {
    LOG(INFO) << "./main <train_fname> <test_fname>";
  }
  return 0;
}
