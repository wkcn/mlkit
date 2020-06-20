#ifndef BOOSTED_TREE_LOSS_H_
#define BOOSTED_TREE_LOSS_H_

#include <algorithm>
#include <cmath>
#include <numeric>

#include "./vec.h"

template <typename T>
inline T clip(T value, T min_value, T max_value) {
  if (value <= min_value) return min_value;
  if (value >= max_value) return max_value;
  return value;
}

struct Loss {
};

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
  inline static T predict(T x) {
    return x;
  }
  template <typename T>
  inline static T estimate(const Vec<T> &Y) {
    return std::accumulate(Y.begin(), Y.end(), T(0)) / Y.size();
  }
};

struct LogisticLoss {
  template <typename T>
  inline static T compute(T x, T y) {
    const T eps = 1e-16;
    const T pred = predict(x);
    return -y * log(std::max(pred, eps)) - \
      (T(1) - y) * log(std::max(T(1) - pred, eps));
  }
  template <typename T>
  inline static T gradient(T x, T y) {
    return predict(x) - y;
  }
  template <typename T>
  inline static T hessian(T x, T y) {
    const T eps = 1e-16;
    const T pred = predict(x);
    return std::max(pred * (T(1) - pred), eps);
  }
  template <typename T>
  inline static T predict(T x) {
    return T(1) / (T(1) + exp(-x));
  }
  template <typename T>
  inline static T estimate(const Vec<T> &Y) {
    const T eps = 1e-16;
    T mean = std::max(std::accumulate(Y.begin(), Y.end(), T(0)) / Y.size(), eps);
    return -log(std::max(T(1) / mean - T(1), eps));
  }
};

#endif
