#include <algorithm>
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
    inline size_t size() const {
      return entries.size();
    }
    Entry& operator[](int i) {
      static Entry begin{0, 0, 0, 0};
      static Entry end{0, 0, 0, 0};
      if (i < 0) return begin;
      if (i >= size()) {
        end.rmin = end.rmax = entries.last().rmax;
        return end;
      }
      return entries[i];
    }
    inline Entry& front() {
      return entries.front();
    }
    inline Entry& back() {
      return entries.back();
    }
  };
  Summary Merge(const Summary &a, const Summary &b) {
    Summary out;
    auto &entries_out = out.entries;
    entries_out.reserve(a.size() + b.size());
    int ai = 0, bi = 0;
    while (ai < a.size() && bi < b.size()) {
      auto &ea = a[ai];
      auto &eb = b[bi];
      if (ea.value == eb.value) {
        entries_out.emplace_back({ea.value, ea.rmin + eb.rmin,
            ea.rmax + eb.rmax, ea.w + eb.w});
        ++ai; ++bi;
      } else if (ea.value < eb.value) {
        // entries_b[bi - 1] < ea.value < entries_b[bi].value
        entries_out.emplace_back({ea.value, ea.rmin + b[bi-1].RMinNext(),
            ea.rmax + eb.RMaxPrev(), ea.w});
        ++ai;
      } else {
        // ea.value > eb.value
        entries_out.emplace_back({eb.value, eb.rmin + a[ai-1].RMinNext(),
            eb.rmax + ea.RMaxPrev(), eb.w});
        ++bi;
      }
    } 
    return out;
  }
  Summary Prune(const Summary &a, RType b, const RType w) {
    Summary out;
    auto &entries_out = out.entries;
    entries_out.reserve(a.size());
    const Entry &front = a.front();
    const Entry &back = a.back();
    int i;
    for (i = 0; i <= b; ++i) {
      RType d = RType(i) * w / b;
      RType _2d = RType(2) * d;
      Entry &e = entries_out[i];
      if (_2d < (front.rmin + front.rmax)) {
        entries_out.emplace_back({front.value, e.rmin, e.rmax, e.w});
      } else break;
    }
    int j = 0;
    for (; i <= b; ++i) {
      RType d = RType(i) * w / b;
      RType _2d = RType(2) * d;
      Entry &e = entries_out[i];

      while (j < a.size() - 1 && !((_2d >= (a[j].rmin + a[j].rmax)) &&
            (_2d < (a[j+1].rmin + a[j+1].rmax)))) {
        ++j;
      }
      if (j > a.size()) break;

      RType x = (_2d < a[i].rmin + a[i].w + a[i+1].rmax - a[i+1].w) ? entries_out[j].value : entries_out[j+1].value;
      entries_out.emplace_back({x, e.rmin, e.rmax, e.w});
    }
    for (; i <= b; ++i) {
      Entry &e = entries_out[i];
      entries_out.emplace_back({back.value, e.rmin, e.rmax, e.w});
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
  return 0;
}
