#ifndef BOOSTED_TREE_TYPE_CONVERT_H_
#define BOOSTED_TREE_TYPE_CONVERT_H_

#include <string>

template <typename T>
T stonum(const std::string &s);

template <>
int stonum(const std::string &s) {
  return std::stoi(s);
}

template <>
float stonum(const std::string &s) {
  return std::stof(s);
}

template <>
double stonum(const std::string &s) {
  return std::stod(s);
}

template <>
std::string stonum(const std::string &s) {
  return s;
}

#endif
