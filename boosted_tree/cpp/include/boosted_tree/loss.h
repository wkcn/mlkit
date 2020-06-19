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
    return std::accumulate(Y.begin(), Y.end(), T(0)) / Y.size();
  }
};

struct LogisticLoss {
  template <typename T>
  inline static T compute(T x, T y) {
    return y * log(T(1) + exp(-x)) + \
      (T(1) - y) * log(T(1) + exp(x));
  }
  template <typename T>
  inline static T gradient(T x, T y) {
    return T(1) / (T(1) + exp(-x)) - y;
  }
  template <typename T>
  inline static T hessian(T x, T y) {
    return T(1) / (exp(-x) + exp(x) + T(2));
  }
  template <typename T>
  inline static T predict(const Vec<T> &Y) {
    T su = std::accumulate(Y.begin(), Y.end(), T(0));
    return log(su) - log(T(Y.size()) - su);
  }
};

#endif
