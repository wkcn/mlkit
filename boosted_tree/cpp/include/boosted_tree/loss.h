#ifndef BOOSTED_TREE_LOSS_H_
#define BOOSTED_TREE_LOSS_H_

#include <algorithm>
#include <cmath>
#include <numeric>

#include "./vec.h"

struct SquareLoss {
  template <typename T>
  inline static T compute(T x, T y) {
    // out = (y - x) ^ 2
    T diff = y - x;
    return diff * diff;
  }
  template <typename T>
  inline static T gradient(T x, T y) {
    // first order gradient
    // out' = 2 * (x - y)
    return 2 * (x - y);
  }
  template <typename T>
  inline static T hessian(T x, T y) {
    // second order gradient
    // out'' = 2
    return 2;
  }
  template <typename T>
  inline static T predict(const Vec<T> &Y) {
    return std::accumulate(Y.begin(), Y.end(), 0) / Y.size();
  }
};

struct LogisticLoss {
  template <typename T>
  inline static T compute(T x, T y) {
    return y * log(1 + exp(-x)) + \
      (1 - y) * log(1 + exp(x));
  }
  template <typename T>
  inline static T gradient(T x, T y) {
    return 1 / (1 + exp(-x)) - y;
  }
  template <typename T>
  inline static T hessian(T x, T y) {
    return 1 / (exp(-x) + exp(x) + 2);
  }
  template <typename T>
  inline static T predict(const Vec<T> &Y) {
    T mean = std::accumulate(Y.begin(), Y.end(), 0) / Y.size();
    return log(mean / (1 - mean));
  }
};

#endif
