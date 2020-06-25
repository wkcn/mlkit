#include <boosted_tree/type_convert.h>

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
