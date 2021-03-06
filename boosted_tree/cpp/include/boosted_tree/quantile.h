#pragma once

#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <utility>
#include <vector>

template <typename DType, typename RType>
class Quantile {
 public:
  struct Entry {
    DType value;  // x
    RType rmin, rmax;
    RType w;
    inline RType RMinNext() const { return rmin + w; }
    inline RType RMaxPrev() const { return rmax - w; }
  };
  struct Summary {
    std::vector<Entry> entries;  // ordered value, no duplicated values
    Summary() {}
    Summary(const Entry &entry) : entries{entry} {};
    Summary(std::vector<std::pair<DType, RType>> &data) {
      if (data.empty()) return;
      std::sort(data.begin(), data.end());
      DType value = data[0].first;
      // wsum (<=), last_wsum(<)
      RType wsum(data[0].second), last_wsum(0);
      entries.reserve(data.size());
      for (int i = 1; i < data.size(); ++i) {
        if (value == data[i].first) {
          wsum += data[i].second;
        } else {
          entries.emplace_back(Entry{value, last_wsum, wsum, wsum - last_wsum});
          value = data[i].first;
          last_wsum = wsum;
          wsum += data[i].second;
        }
      }
      entries.emplace_back(Entry{value, last_wsum, wsum, wsum - last_wsum});
    }
    inline size_t size() const { return entries.size(); }
    inline bool empty() const { return entries.empty(); }
    const Entry &operator[](int i) const {
      static Entry begin{DType(0), RType(0), RType(0), RType(0)};
      static Entry end{DType(0), RType(0), RType(0), RType(0)};
      if (i < 0) return begin;
      if (i >= size()) {
        end.rmin = end.rmax = entries.back().rmax;
        return end;
      }
      return entries[i];
    }
    inline const Entry &front() const { return entries.front(); }
    inline const Entry &back() const { return entries.back(); }
  };
  static Summary Merge(const Summary &a, const Summary &b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    Summary out;
    auto &entries_out = out.entries;
    entries_out.reserve(a.size() + b.size());
    int ai = 0, bi = 0;
    while (ai < a.size() && bi < b.size()) {
      const auto &ea = a[ai];
      const auto &eb = b[bi];
      if (ea.value == eb.value) {
        AccumulateEntry(entries_out, Entry{ea.value, ea.rmin + eb.rmin,
                                           ea.rmax + eb.rmax, ea.w + eb.w});
        ++ai;
        ++bi;
      } else if (ea.value < eb.value) {
        // entries_b[bi - 1] < ea.value < entries_b[bi].value
        AccumulateEntry(entries_out,
                        Entry{ea.value, ea.rmin + b[bi - 1].RMinNext(),
                              ea.rmax + eb.RMaxPrev(), ea.w});
        ++ai;
      } else {
        // ea.value > eb.value
        AccumulateEntry(entries_out,
                        Entry{eb.value, eb.rmin + a[ai - 1].RMinNext(),
                              eb.rmax + ea.RMaxPrev(), eb.w});
        ++bi;
      }
    }
    while (ai < a.size()) {
      const auto &ea = a[ai++];
      RType r = b.entries.back().rmax;
      AccumulateEntry(entries_out,
                      Entry{ea.value, ea.rmin + r, ea.rmax + r, ea.w});
    }
    while (bi < b.size()) {
      const auto &eb = b[bi++];
      RType r = a.entries.back().rmax;
      AccumulateEntry(entries_out,
                      Entry{eb.value, eb.rmin + r, eb.rmax + r, eb.w});
    }
    return out;
  }
  static Summary Prune(const Summary &a, const int b) {
    Summary out;
    auto &entries_out = out.entries;
    entries_out.reserve(b + 1);
    const Entry &front = a.front();
    const Entry &back = a.back();
    const RType wsum = a.back().rmax;
    int i;
    for (i = 0; i <= b; ++i) {
      const float d = float(i) * wsum / b;
      const float _2d = float(2) * d;
      // _2d may be duplicated
      if (_2d < (front.rmin + front.rmax)) {
        // x1
        AppendUniqueEntry(entries_out, front);
      } else
        break;
    }
    // _2d >= (front.rmin + front.rmax)
    int j = 0;
    for (; i <= b; ++i) {
      const float d = float(i) * wsum / b;
      const float _2d = float(2) * d;

      // find j such that
      // _2d >= a[j].rmin + a[j].rmax and _2d < (a[j+1].rmin + a[j+1].rmax)
      while (j < a.size() - 1 && (!(_2d < a[j + 1].rmin + a[j + 1].rmax))) {
        ++j;
      }
      if (j >= a.size() - 1) break;

      if (_2d < a[j].RMinNext() + a[j + 1].RMaxPrev()) {
        AppendUniqueEntry(entries_out, a[j]);
      } else {
        AppendUniqueEntry(entries_out, a[j + 1]);
      }
    }
    // _2d >= back.rmin + back.rmax
    for (; i <= b; ++i) {
      // x_k
      AppendUniqueEntry(entries_out, back);
    }
    return out;
  }
  static void AccumulateEntry(std::vector<Entry> &entries_out,
                              const Entry &entry) {
    if (entries_out.empty()) {
      entries_out.push_back(entry);
    } else {
      Entry &last = entries_out.back();
      if (last.value != entry.value) {
        entries_out.push_back(entry);
      } else {
        last.rmin += entry.rmin;
        last.rmax += entry.rmax;
        last.w += entry.w;
      }
    }
  }
  static void AppendUniqueEntry(std::vector<Entry> &entries_out,
                                const Entry &entry) {
    if (entries_out.empty()) {
      entries_out.push_back(entry);
    } else {
      if (entries_out.back().value != entry.value) {
        entries_out.push_back(entry);
      }
    }
  }
};
