#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "kmeans.h"
#include "vec.h"
using namespace std;

vector<Vec<float>> ReadData(const string &filename) {
  ifstream fin(filename);
  int rows, cols;
  fin >> rows >> cols;
  vector<Vec<float>> res;
  for (int r = 0; r < rows; ++r) {
    Vec<float> v(cols);
    for (int c = 0; c < cols; ++c) {
      fin >> v[c];
    }
    res.emplace_back(std::move(v));
  }
  return res;
}

int main() {
  srand(time(0));

  vector<Vec<float>> data = ReadData("./data.txt");
  cout << "Rows: " << data.size() << endl;
  cout << "Cols: " << data[0].size() << endl;
  cout << endl;

  const int K = 3;
  int max_iter = 10;
  KMeansResult<float> res = KMeans(data, K, max_iter);

  cout << "Centers: " << endl;
  for (auto &v : res.centers) {
    cout << v << endl;
  }

  cout << "Labels: " << endl;
  bool first = true;
  for (int v : res.labels) {
    if (!first) cout << ' ';
    first = false;
    cout << v;
  }
  cout << endl;

  cout << "Inertia: " << endl;
  cout << res.inertia << endl;
  return 0;
}
