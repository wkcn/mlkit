#ifndef BOOSTED_TREE_OBJECTIVE_H_
#define BOOSTED_TREE_OBJECTIVE_H_

#include <algorithm>
#include <cmath>
#include <numeric>

#include "./registry.h"
#include "./vec.h"

template <typename T>
inline T clip(T value, T min_value, T max_value) {
  if (value <= min_value) return min_value;
  if (value >= max_value) return max_value;
  return value;
}

template <typename T>
class Objective {
 public:
  virtual inline T compute(T x, T y) = 0;
  virtual inline T gradient(T x, T y) = 0;
  virtual inline T hessian(T x, T y) = 0;
  virtual inline T predict(T x) = 0;
  virtual inline T estimate(const Vec<T> &Y) = 0;
};
REGISTRY_ENABLE(Objective<float>);

template <typename T>
class SquareLoss : public Objective<T> {
 public:
  inline T compute(T x, T y) {
    // out = (y - x) ^ 2
    T diff = y - x;
    return diff * diff;
  }
  inline T gradient(T x, T y) {
    // first order gradient
    // out' = 2 * (x - y)
    return 2 * (x - y);
  }
  inline T hessian(T x, T y) {
    // second order gradient
    // out'' = 2
    return 2;
  }
  inline T predict(T x) { return x; }
  inline T estimate(const Vec<T> &Y) { return (T)Y.sum() / Y.size(); }
};

template <typename T>
class LogisticLoss : public Objective<T> {
 public:
  inline T compute(T x, T y) {
    const T eps = 1e-16;
    const T pred = predict(x);
    return -y * log(std::max(pred, eps)) -
           (T(1) - y) * log(std::max(T(1) - pred, eps));
  }
  inline T gradient(T x, T y) { return predict(x) - y; }
  inline T hessian(T x, T y) {
    const T eps = 1e-16;
    const T pred = predict(x);
    return std::max(pred * (T(1) - pred), eps);
  }
  inline T predict(T x) { return T(1) / (T(1) + exp(-x)); }
  inline T estimate(const Vec<T> &Y) {
    const T eps = 1e-16;
    T mean = std::max((T)Y.sum() / Y.size(), eps);
    return -log(std::max(T(1) / mean - T(1), eps));
  }
};

class ObjectiveRegistry {
 public:
  ObjectiveRegistry() {
    typedef float T;
    Registry<Objective<T>>::Register("reg:linear", new SquareLoss<T>());
    Registry<Objective<T>>::Register("binary:logistic", new LogisticLoss<T>());
  }
};
static ObjectiveRegistry registry_;

#endif
