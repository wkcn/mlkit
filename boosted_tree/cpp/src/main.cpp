#include <vector>
#include <string>

#include <boosted_tree/boosted_tree.h>
#include <boosted_tree/csr_matrix.h>
#include <boosted_tree/io.h>
#include <boosted_tree/logging.h>

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
  } else {
    LOG(INFO) << "Please pass the argument `filename`";
  }
  bst.train(X, Y);
  return 0;
}
