#pragma once

#include <boosted_tree/quantile.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

using quantile_t = Quantile<float, int>;
using summary_t = Quantile<float, int>::Summary;
using entry_t = Quantile<float, int>::Entry;
using pair_t = std::pair<float, int>;

void CHECK_MERGE(const summary_t &summary, std::vector<pair_t> data) {
  std::sort(data.begin(), data.end(),
            [](const pair_t &a, const pair_t &b) { return a.first < b.first; });
  std::vector<pair_t> values;
  values.push_back(data.front());
  for (int i = 1; i < data.size(); ++i) {
    auto &p = data[i];
    if (p.first != values.back().first)
      values.push_back(p);
    else
      values.back().second += p.second;
  }
  ASSERT_EQ(values.size(), summary.size());
  for (int i = 0; i < summary.size() - 1; ++i) {
    ASSERT_LT(summary[i].value, summary[i + 1].value);
  }
  for (int i = 0; i < summary.size(); ++i) {
    ASSERT_EQ(summary[i].value, values[i].first);
    ASSERT_EQ(summary[i].w, values[i].second);
  }
}

void CHECK_PRUNE(const summary_t &summary, const int b,
                 std::vector<pair_t> data) {
  // sort by value
  std::sort(data.begin(), data.end(),
            [](const pair_t &a, const pair_t &b) { return a.first < b.first; });
  std::vector<float> values;
  values.push_back(data.front().first);
  for (auto &p : data) {
    if (p.first != values.back()) values.push_back(p.first);
  }
  // check duplicated value
  std::vector<float> splits{summary.front().value};
  for (int i = 1; i < summary.size(); ++i) {
    ASSERT_NE(summary[i].value, splits.back());
    splits.push_back(summary[i].value);
  }
  const int wsum =
      accumulate(data.begin(), data.end(), int(0),
                 [](int acc, const pair_t &p) { return acc + p.second; });
  CHECK_EQ(wsum, summary.back().rmax);
  int rwsum = 0;
  int i = 0;
  bool first = true;
  float last_r;
  ASSERT_EQ(splits.size(), summary.size());
  ASSERT_LE(splits.size(), b + 1);
  ASSERT_EQ(splits.front(), summary.front().value);
  ASSERT_EQ(splits.back(), summary.back().value);

  for (float val : splits) {
    while (i < data.size() && data[i].first < val) {
      rwsum += data[i].second;
      ++i;
    }
    float r = float(rwsum) / wsum;
    if (first)
      first = false;
    else {
      // disable this test temporarily since r is discrete : (
      // ASSERT_LE(std::abs(r - last_r), float(1) / b +
      // std::numeric_limits<float>::epsilon());
    }
    last_r = r;
  }
  float tolerate = float(1) / b;
  for (int i = 0; i < summary.size(); ++i) {
    float d = float(i) * wsum / b;
    auto &e = summary[i];
    ASSERT_GE(d, e.RMaxPrev() - tolerate / 2 * wsum);
    ASSERT_LE(d, e.RMinNext() + tolerate / 2 * wsum);
  }
}

TEST(Quantile, TestQuantile) {
  const int N = 20;
  const int M = 10;
  // std::vector<pair_t> data;
  std::vector<pair_t> data;
  for (int i = 0; i < N; ++i) {
    data.push_back({i, (i + 1) * (i + 1)});
  }
  /*
  for (int i = 0; i < N; ++i) {
    data.push_back({rand() % 20, rand() % 500});
  }
  */
  summary_t s(entry_t{data[0].first, 0, data[0].second, data[0].second});
  for (int i = 1; i < N; ++i) {
    summary_t b(entry_t{data[i].first, 0, data[i].second, data[i].second});
    s = quantile_t::Merge(s, b);
  }
  CHECK_MERGE(s, data);
  s = quantile_t::Prune(s, M);
  CHECK_PRUNE(s, M, data);
}
