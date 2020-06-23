#ifndef _BOOSTED_TREE_ARRAY_H_
#define _BOOSTED_TREE_ARRAY_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <initializer_list>
#include <iostream>

#define DEF_ARRAY_OP_SCALAR_FUNC(op)     \
  Array<T, N> &operator op(const T &b) { \
    for (int i = 0; i < N; ++i) {        \
      (*this)[i] op b;                   \
    }                                    \
    return *this;                        \
  }

#define DEF_ARRAY_OP_ARRAY_FUNC(op)                \
  Array<T, N> &operator op(const Array<T, N> &b) { \
    for (int i = 0; i < N; ++i) {                  \
      (*this)[i] op b[i];                          \
    }                                              \
    return *this;                                  \
  }

template <typename T, size_t N>
class Array : public std::array<T, N> {
 public:
  Array() : std::array<T, N>() {}
  Array(std::initializer_list<T> il) {
    int i = 0;
    for (const auto &v : il) {
      (*this)[i] = v;
    }
  }
  template <class InputIterator>
  Array(InputIterator first, InputIterator last)
      : std::array<T, N>(first, last) {}

 public:
  // scalar
  DEF_ARRAY_OP_SCALAR_FUNC(+=)
  DEF_ARRAY_OP_SCALAR_FUNC(-=)
  DEF_ARRAY_OP_SCALAR_FUNC(*=)
  DEF_ARRAY_OP_SCALAR_FUNC(/=)
  // array
  DEF_ARRAY_OP_ARRAY_FUNC(+=)
  DEF_ARRAY_OP_ARRAY_FUNC(-=)
  DEF_ARRAY_OP_ARRAY_FUNC(*=)
  DEF_ARRAY_OP_ARRAY_FUNC(/=)
};

#define DEF_ARRAY_OP_SCALAR_BINARY_FUNC(op, aop)              \
  template <typename T, size_t N>                             \
  Array<T, N> operator op(const Array<T, N> &a, const T &b) { \
    Array<T, N> c = a;                                        \
    c aop b;                                                  \
    return c;                                                 \
  }

DEF_ARRAY_OP_SCALAR_BINARY_FUNC(+, +=)
DEF_ARRAY_OP_SCALAR_BINARY_FUNC(-, -=)
DEF_ARRAY_OP_SCALAR_BINARY_FUNC(*, *=)
DEF_ARRAY_OP_SCALAR_BINARY_FUNC(/, /=)

#define DEF_ARRAY_OP_ARRAY_BINARY_FUNC(op, aop)                         \
  template <typename T, size_t N>                                       \
  Array<T, N> operator op(const Array<T, N> &a, const Array<T, N> &b) { \
    Array<T, N> c = a;                                                  \
    c aop b;                                                            \
    return c;                                                           \
  }

DEF_ARRAY_OP_ARRAY_BINARY_FUNC(+, +=)
DEF_ARRAY_OP_ARRAY_BINARY_FUNC(-, -=)
DEF_ARRAY_OP_ARRAY_BINARY_FUNC(*, *=)
DEF_ARRAY_OP_ARRAY_BINARY_FUNC(/, /=)

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &os, const Array<T, N> &v) {
  if (N > 0) os << v[0];
  for (int i = 1; i < N; ++i) {
    os << ' ' << v[i];
  }
  return os;
}

template <typename srcT, typename dstT, size_t N>
Array<dstT, N> AsType(const Array<srcT, N> &src) {
  if (N <= 0) return {};
  Array<dstT, N> a;
  for (int i = 0; i < N; ++i) a[i] = src[i];
  return a;
}

#endif
