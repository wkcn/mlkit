#pragma once
// Reference: https://github.com/dmlc/dmlc-core/blob/master/include/dmlc/registry.h
#include <map>
#include <memory>
#include <mutex>

template <typename Entry>
class Registry {
public:
  static void Register(const std::string &name, Entry *entry);
  static Entry* Find(const std::string &name);
  static Registry<Entry>& Get();
private:
  std::unordered_map<std::string, std::unique_ptr<Entry> > fmap_;
  std::mutex mtx_;
};

template<typename Entry>
void Registry<Entry>::Register(const std::string &name, Entry *entry) {
  Registry<Entry> &self = Get();
  std::lock_guard<std::mutex> lck(self.mtx_);
  self.fmap_[name].reset(entry);
}

template<typename Entry>
Entry* Registry<Entry>::Find(const std::string &name) {
  Registry<Entry> &self = Get();
  auto p = self.fmap_.find(name);
  return p != self.fmap_.end() ? (p->second).get() : nullptr;
}

#define REGISTRY_ENABLE(Entry) \
  template<> \
  Registry<Entry>& Registry<Entry>::Get() { \
    static Registry<Entry> inst; \
    return inst; \
  }
