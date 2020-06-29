#ifndef _BOOSTED_TREE_VEC_H_
#define _BOOSTED_TREE_VEC_H_

#include <cassert>
#include <initializer_list>
#include <iostream>
#include <valarray>
#include <vector>

#define DEF_VEC_OP_SCALAR_FUNC(op)  \
  Vec<T> &operator op(const T &b) { \
    const int N = this->size();     \
    for (int i = 0; i < N; ++i) {   \
      (*this)[i] op b;              \
    }                               \
    return *this;                   \
  }

#define DEF_VEC_OP_VEC_FUNC(op)          \
  Vec<T> &operator op(const Vec<T> &b) { \
    assert(this->size() == b.size());    \
    const int N = this->size();          \
    for (int i = 0; i < N; ++i) {        \
      (*this)[i] op b[i];                \
    }                                    \
    return *this;                        \
  }

template <typename T>
class Vec : public std::valarray<T> {
 public:
  Vec() : std::valarray<T>() {}
  Vec(size_t n) : std::valarray<T>(n) {}
  Vec(std::initializer_list<T> il) : std::valarray<T>(il) {}
  template <class InputIterator>
  Vec(InputIterator first, InputIterator last) {
    const size_t n = std::distance(first, last);
    this->resize(n);
    int i = 0;
    for (auto p = first; p != last; ++p, ++i) {
      (*this)[i] = *p;
    }
  }
  Vec(const std::vector<T> &data) : Vec<T>(std::begin(data), std::end(data)) {}
  Vec(Vec &) = default;
  Vec(const Vec &) = default;
  Vec(Vec &&) = default;
  Vec& operator=(const Vec &) = default;
  Vec& operator=(Vec &&) = default;
  T* data() {return &((*this)[0]);}
  const T* data() const {return &((*this)[0]);}

 public:
  // scalar
  DEF_VEC_OP_SCALAR_FUNC(+=)
  DEF_VEC_OP_SCALAR_FUNC(-=)
  DEF_VEC_OP_SCALAR_FUNC(*=)
  DEF_VEC_OP_SCALAR_FUNC(/=)
  // vec
  DEF_VEC_OP_VEC_FUNC(+=)
  DEF_VEC_OP_VEC_FUNC(-=)
  DEF_VEC_OP_VEC_FUNC(*=)
  DEF_VEC_OP_VEC_FUNC(/=)
};

#include "logging.h"
template <typename T, typename VT>
bool operator==(const Vec<T> &a, const VT &b) {
  auto pa = std::begin(a);
  auto pb = std::begin(b);
  const auto end_a = std::end(a);
  const auto end_b = std::end(b);
  for (; pa != end_a && pb != end_b; ++pa, ++pb) {
    if (*pa != *pb) return false;
  }
  return pa == end_a && pb == end_b;
}

#define DEF_VEC_OP_SCALAR_BINARY_FUNC(op, aop)      \
  template <typename T>                             \
  Vec<T> operator op(const Vec<T> &a, const T &b) { \
    Vec<T> c = a;                                   \
    c aop b;                                        \
    return c;                                       \
  }                                                 \
  template <typename T>                             \
  Vec<T> operator op(const T &a, const Vec<T> &b) { \
    Vec<T> c = b;                                   \
    c aop a;                                        \
    return c;                                       \
  }

DEF_VEC_OP_SCALAR_BINARY_FUNC(+, +=)
DEF_VEC_OP_SCALAR_BINARY_FUNC(-, -=)
DEF_VEC_OP_SCALAR_BINARY_FUNC(*, *=)
DEF_VEC_OP_SCALAR_BINARY_FUNC(/, /=)

#define DEF_VEC_OP_VEC_BINARY_FUNC(op, aop)              \
  template <typename T>                                  \
  Vec<T> operator op(const Vec<T> &a, const Vec<T> &b) { \
    assert(a.size() == b.size());                        \
    Vec<T> c = a;                                        \
    c aop b;                                             \
    return c;                                            \
  }

DEF_VEC_OP_VEC_BINARY_FUNC(+, +=)
DEF_VEC_OP_VEC_BINARY_FUNC(-, -=)
DEF_VEC_OP_VEC_BINARY_FUNC(*, *=)
DEF_VEC_OP_VEC_BINARY_FUNC(/, /=)

template <typename T>
std::ostream &operator<<(std::ostream &os, const Vec<T> &v) {
  const int N = v.size();
  if (N > 0) os << v[0];
  for (int i = 1; i < N; ++i) {
    os << ' ' << v[i];
  }
  return os;
}

template <typename T>
T Sum(const Vec<T> &v) {
  T a = 0;
  for (const T &x : v) a += x;
  return a;
}

template <typename T>
T Mean(const Vec<T> &v) {
  return Sum(v) / v.size();
}

template <typename T>
T Dot(const Vec<T> &a, const Vec<T> &b) {
  return Sum(a * b);
}

template <typename srcT, typename dstT>
Vec<dstT> AsType(const Vec<srcT> &src) {
  const int N = src.size();
  if (N <= 0) return {};
  Vec<dstT> a;
  for (int i = 0; i < N; ++i) a[i] = src[i];
  return a;
}

#endif
