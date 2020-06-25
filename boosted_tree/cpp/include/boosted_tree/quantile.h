#pragma once

#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <vector>

template <typename DType, typename RType>
class Quantile {
public:
  struct Entry {
    DType value;  // x
    RType rmin, rmax;
    RType w;
    inline RType RMinNext() const {
      return rmin + w;
    }
    inline RType RMaxPrev() const {
      return rmax - w;
    } 
  };
  struct Summary {
    std::vector<Entry> entries; // ordered value, no duplicated values
    Summary() {}
    Summary(const Entry& entry) : entries{entry} {};
    inline size_t size() const {
      return entries.size();
    }
    const Entry& operator[](int i) const {
      static Entry begin{0, 0, 0, 0};
      static Entry end{0, 0, 0, 0};
      if (i < 0) return begin;
      if (i >= size()) {
        end.rmin = end.rmax = entries.back().rmax;
        return end;
      }
      return entries[i];
    }
    inline const Entry& front() const {
      return entries.front();
    }
    inline const Entry& back() const {
      return entries.back();
    }
  };
  static Summary Merge(const Summary &a, const Summary &b) {
    Summary out;
    auto &entries_out = out.entries;
    entries_out.reserve(a.size() + b.size());
    int ai = 0, bi = 0;
    while (ai < a.size() && bi < b.size()) {
      const auto &ea = a[ai];
      const auto &eb = b[bi];
      if (ea.value == eb.value) {
        entries_out.emplace_back(Entry{ea.value, ea.rmin + eb.rmin,
            ea.rmax + eb.rmax, ea.w + eb.w});
        ++ai; ++bi;
      } else if (ea.value < eb.value) {
        // entries_b[bi - 1] < ea.value < entries_b[bi].value
        entries_out.emplace_back(Entry{ea.value, ea.rmin + b[bi-1].RMinNext(),
            ea.rmax + eb.RMaxPrev(), ea.w});
        ++ai;
      } else {
        // ea.value > eb.value
        entries_out.emplace_back(Entry{eb.value, eb.rmin + a[ai-1].RMinNext(),
            eb.rmax + ea.RMaxPrev(), eb.w});
        ++bi;
      }
    }
    while (ai < a.size()) {
      const auto &ea = a[ai++];
      RType r = b.entries.back().rmax;
      entries_out.emplace_back(Entry{ea.value, ea.rmin + r, ea.rmax + r, ea.w});
    }
    while (bi < b.size()) {
      const auto &eb = b[bi++];
      RType r = a.entries.back().rmax;
      entries_out.emplace_back(Entry{eb.value, eb.rmin + r, eb.rmax + r, eb.w});
    }
    return out;
  }
  static Summary Prune(const Summary &a, RType b) {
    Summary out;
    auto &entries_out = out.entries;
    entries_out.reserve(b + 1);
    const Entry &front = a.front();
    const Entry &back = a.back();
    RType wsum = std::accumulate(a.entries.begin(), a.entries.end(), RType(0), [](RType acc, const Entry& e) -> RType {
        return acc + e.w;
    });
    int i;
    for (i = 0; i <= b; ++i) {
      const float d = float(i) * wsum / b;
      const float _2d = float(2) * d;
      if (_2d < (front.rmin + front.rmax)) {
        // x1
        AppendEntry(entries_out, front);
      } else break;
    }
    int j = 0;
    for (; i <= b; ++i) {
      const float d = float(i) * wsum / b;
      const float _2d = float(2) * d;
      const Entry &e = a[i];

      while (j < a.size() - 1 && ((_2d < a[j].rmin + a[j].rmax) ||
            (_2d >= a[j+1].rmin + a[j+1].rmax))) {
        ++j;
      }
      if (j >= a.size() - 1) break;

      if (_2d < a[j].RMinNext() + a[j+1].RMaxPrev()) {
        const Entry &e = a[j];
        AppendEntry(entries_out, e);
      } else {
        const Entry &e = a[j + 1];
        AppendEntry(entries_out, e);
      }
    }
    for (; i <= b; ++i) {
      // x_k
      AppendEntry(entries_out, back);
    }
    return out;
  }
  static void AppendEntry(std::vector<Entry> &entries_out, const Entry &entry) {
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
};
