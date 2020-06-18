#include <vector>

#include <boosted_tree/boosted_tree.h>
#include <boosted_tree/csr_matrix.h>

int main(int argc, char **argv) {
  BoostedTree bst;
  bst.train(CSRMatrix<float>(0, 0), {});
  return 0;
}
