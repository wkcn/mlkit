#include <algorithm>
#include <numeric>
#include <vector>

#include <iostream>
using namespace std;

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
        entries_out.emplace_back(Entry{front.value, front.rmin, front.rmax, front.w});
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
        entries_out.emplace_back(Entry{e.value, e.rmin, e.rmax, e.w});
      } else {
        const Entry &e = a[j + 1];
        entries_out.emplace_back(Entry{e.value, e.rmin, e.rmax, e.w});
      }
    }
    for (; i <= b; ++i) {
      // x_k
      entries_out.emplace_back(Entry{back.value, back.rmin, back.rmax, back.w});
    }
    return out;
  }
private:
  inline DType Query(const Summary &a, RType d) {
    const Entry &front = a.front();
    RType _2d = RType(2) * d;
    if (_2d < (front.rmin + front.rmax)) return front.value;
    const Entry &back = a.back();
    if (_2d >= (back.rmin + back.rmax)) return back.value;
    for (int i = 0; i < a.size() - 1; ++i) {
      if ((_2d >= (a[i].rmin + a[i].rmax)) &&
            (_2d < (a[i+1].rmin + a[i+1].rmax))) {
        if (_2d < a[i].rmin + a[i].w + a[i+1].rmax - a[i+1].w) return a[i].value;
        return a[i+1].value;
      }
    }
    throw "TODO FATAL";
  } 
};

int main() {
  Quantile<float, int> q;
  std::vector<int> vs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  using quantile_t = Quantile<float, int>;
  using summary_t = Quantile<float, int>::Summary;
  using entry_t = Quantile<float, int>::Entry;
  std::vector<summary_t> summaries;
  for (int x : vs) {
    summaries.emplace_back(entry_t{float(x), 0, 1, 1});
  }
  summary_t s = summaries[0];
  for (int i = 1; i < summaries.size(); ++i) {
    s = quantile_t::Merge(s, summaries[i]);
  }
  s = quantile_t::Prune(s, 3);
  for (auto e : s.entries) {
    cout << e.value << ": " << e.rmin << ", " << e.rmax << endl;
  }
  return 0;
}
