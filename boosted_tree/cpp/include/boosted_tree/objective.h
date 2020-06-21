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

class ObjectiveBase {
public:
  typedef float T;
  virtual inline T compute(T x, T y) = 0;
  virtual inline T gradient(T x, T y) = 0;
  virtual inline T hessian(T x, T y) = 0;
  virtual inline T predict(T x) = 0;
  virtual inline T estimate(const Vec<T> &Y) = 0;
};
REGISTRY_ENABLE(ObjectiveBase);

template <typename Loss>
class Objective : public ObjectiveBase {
public:
  typedef float T;
  inline T compute(T x, T y) {
    return Loss::compute(x, y);
  }
  inline T gradient(T x, T y) {
    return Loss::gradient(x, y);
  }
  inline T hessian(T x, T y) {
    return Loss::hessian(x, y);
  }
  inline T predict(T x) {
    return Loss::predict(x);
  }
  inline T estimate(const Vec<T> &Y) {
    return Loss::estimate(Y);
  }
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

class ObjectiveBaseRegistry {
public:
  ObjectiveBaseRegistry() {
    Registry<ObjectiveBase>::Register("reg:linear", new Objective<SquareLoss>());
    Registry<ObjectiveBase>::Register("binary:logistic", new Objective<LogisticLoss>());
  }
};
static ObjectiveBaseRegistry registry_; 

#endif