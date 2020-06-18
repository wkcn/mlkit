#ifndef _KMEANS_H_
#define _KMEANS_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "vec.h"

template <typename T>
struct KMeansResult {
  std::vector<Vec<T>> centers;
  std::vector<int> labels;
  T inertia;
};

template <typename T>
void Shuffle(std::vector<T> &arr) {
  const int N = arr.size();
  if (N <= 0) return;
  for (int i = 0; i < N; ++i) {
    int r = rand() % N;
    std::swap(arr[i], arr[r]);
  }
}

template <typename T>
std::vector<T> Range(const int N) {
  if (N <= 0) return {};
  std::vector<T> arr(N);
  for (int i = 0; i < N; ++i) arr[i] = i;
  return arr;
}

template <typename T>
KMeansResult<T> KMeansImpl(const std::vector<Vec<T>> &data, const int K,
                           const int max_iter, const T tol) {
  const int N = data.size();
  const int NDIM = data[0].size();
  std::vector<int> inds = Range<int>(N);
  Shuffle(inds);
  std::vector<Vec<T>> centers;
  for (int c = 0; c < K; ++c) {
    centers.push_back(data[inds[c]]);
  }
  std::vector<int> labels(N, -1);
  T inertia = 0;
  for (int iter = 0; iter <= max_iter; ++iter) {
    // compute distance
    std::vector<std::vector<int>> clusters(K);
    bool changed = false;
    T new_inertia = 0;
    for (int i = 0; i < N; ++i) {
      const Vec<T> &v = data[i];
      T best_d2;
      int best_c;
      for (int c = 0; c < K; ++c) {
        const Vec<T> &center = centers[c];
        Vec<T> diff = v - center;
        // square sum
        T d2 = Sum(diff * diff);
        if (c == 0 || d2 < best_d2) {
          best_d2 = d2;
          best_c = c;
        }
      }
      new_inertia += best_d2;
      if (labels[i] != best_c) changed = true;
      labels[i] = best_c;
      clusters[best_c].push_back(i);
    }
    T old_inertia = inertia;
    inertia = new_inertia;
    if (!changed) break;
    if (abs(new_inertia - old_inertia) < tol) break;
    if (iter < max_iter) {
      // compute new centers
      for (int c = 0; c < K; ++c) {
        Vec<T> &center = centers[c];
        center.Fill(0);
        for (int i : clusters[c]) {
          center += data[i];
        }
        center /= clusters[c].size();
      }
    }
  }
  KMeansResult<T> res;
  res.centers = std::move(centers);
  res.labels = std::move(labels);
  res.inertia = inertia;
  return res;
}

template <typename T>
KMeansResult<T> KMeans(const std::vector<Vec<T>> &data, const int K,
                       const int max_iter, const T tol = 1e-4,
                       const int n_init = 1) {
  assert(n_init > 0);
  assert(max_iter > 0);
  assert(K > 0);
  assert(data.size() >= K);
  bool first = true;
  KMeansResult<T> res;
  for (int i = 0; i < n_init; ++i) {
    KMeansResult<T> r = KMeansImpl(data, K, max_iter, tol);
    if (first || r.inertia < res.inertia) {
      res = std::move(r);
      first = false;
    }
  }
  return res;
}

#endif
