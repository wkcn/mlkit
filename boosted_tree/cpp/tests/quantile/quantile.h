#pragma once

#include <boosted_tree/quantile.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

using quantile_t = Quantile<float, int>;
using summary_t = Quantile<float, int>::Summary;
using entry_t = Quantile<float, int>::Entry;
using pair_t = std::pair<float, int>;

void CHECK_ORDER(const summary_t &summary) {
  for (int i = 0; i < summary.size() - 1; ++i) {
    ASSERT_LT(summary[i].value, summary[i + 1].value);
  }
}

void CHECK_ERROR(const summary_t &summary, const int b, std::vector<pair_t> data) {
  // sort by value
  std::sort(data.begin(), data.end(), [](const pair_t &a, const pair_t &b) {
      return a.first < b.first;
  });
  std::vector<float> values;
  values.push_back(data.front().first);
  for (auto &p : data) {
    if (p.first != values.back())
      values.push_back(p.first);
  }
  std::vector<float> splits;
  splits.push_back(summary.front().value);
  for (int i = 1; i < summary.size(); ++i) {
    if (summary[i].value != splits.back()) {
      CHECK_LT(splits.back(), summary[i].value);
      splits.push_back(summary[i].value);
    }
  }
  const int wsum = accumulate(data.begin(), data.end(), int(0), [](int acc, const pair_t &p) {
      return acc + p.second;
  });
  int rwsum = 0;
  int i = 0;
  bool first = true;
  float last_r;
  for (float val : splits) {
    while (i < data.size() && data[i].first <= val) {
      rwsum += data[i].second;
      ++i;
    }
    float r = float(rwsum) / wsum;
    if (first) first = false;
    else {
      CHECK_LT(std::abs(r - last_r), float(1) / b);
    }
    last_r = r;
  }
}

TEST(Quantile, TestOrdered) {
  const int N = 100;
  const int M = 10;
  std::vector<pair_t > data;
  for (int i = 0; i < N; ++i) {
    data.push_back({rand() % 20, rand() % 500 - 1000});
  }
  summary_t s(entry_t{data[0].first, 0, data[0].second, data[0].second});
  for (int i = 1; i < N; ++i) {
    summary_t b(entry_t{data[0].first, 0, data[0].second, data[0].second});
    s = quantile_t::Merge(s, b);
  }
  CHECK_ORDER(s);
  s = quantile_t::Prune(s, M);
  CHECK_ORDER(s);
  CHECK_ERROR(s, M, data);
}
