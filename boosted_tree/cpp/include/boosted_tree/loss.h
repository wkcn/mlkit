#ifndef BOOSTED_TREE_LOSS_H_
#define BOOSTED_TREE_LOSS_H_

#include <cmath>

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
};

#endif
