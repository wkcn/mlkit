#ifndef _BOOSTED_TREE_VEC_H_
#define _BOOSTED_TREE_VEC_H_

#include <cassert>
#include <initializer_list>
#include <iostream>
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
    assert(this->size() == b.size());   \
    const int N = this->size();          \
    for (int i = 0; i < N; ++i) {        \
      (*this)[i] op b[i];                \
    }                                    \
    return *this;                        \
  }

template <typename T>
class Vec : public std::vector<T> {
 public:
  Vec() : std::vector<T>() {}
  Vec(size_t n) : std::vector<T>(n) {}
  Vec(std::initializer_list<T> il) : std::vector<T>(il) {}
  template <class InputIterator>
  Vec(InputIterator first, InputIterator last) : std::vector<T>(first, last) {}
  Vec(const Vec &x) : std::vector<T>(x) {}
  Vec(const Vec &&x) : std::vector<T>(x) {}

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
 public:
  void Fill(T val) {
    const int N = this->size();
    for (int i = 0; i < N; ++i) {
      (*this)[i] = val;
    }
  }
};

#define DEF_VEC_OP_SCALAR_BINARY_FUNC(op, aop)      \
  template <typename T>                             \
  Vec<T> operator op(const Vec<T> &a, const T &b) { \
    Vec<T> c = a;                                   \
    c aop b;                                        \
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
