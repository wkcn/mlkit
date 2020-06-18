#ifndef BOOSTED_TREE_IO_H_
#define BOOSTED_TREE_IO_H_

#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <utility>

#include "./csr_matrix.h"
#include "./logging.h"

std::pair<CSRMatrix<float>, std::vector<float> > ReadLibSVMFile(
    const std::string &filename) {
  std::ifstream fin(filename);
  if (!fin.is_open()) {
    LOG(INFO) << "Open file " << filename << " fail! :(";
    return {CSRMatrix<float>(0, 0), {}};
  }
  std::string buf;

  std::vector<float> labels;
  std::vector<dim_t> row;
  std::vector<dim_t> col;
  std::vector<float> data;
  dim_t rows = 0, cols = 0;
  while (getline(fin, buf)) {
    std::stringstream ss;
    ss << buf;
    float label; ss >> label;
    labels.push_back(label);
    std::string data_str;
    while (ss >> data_str) {
      int i;
      for (i = 0; i < data_str.size(); ++i) {
        if (data_str[i] == ':') break;
      }

      dim_t c = stoi(data_str.substr(0, i));
      float v = stof(data_str.substr(i + 1));
      cols = std::max(cols, c + 1);

      row.push_back(rows);
      col.push_back(c);
      data.push_back(v);
    }
    ++rows;
  }

  TEST_EQ(rows, labels.size());
  CSRMatrix<float> smat(rows, cols);
  smat.reset(row, col, data);
  return {smat, labels};
}

#endif
