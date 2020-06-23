#ifndef MOBULA_INC_LOGGING_H_
#define MOBULA_INC_LOGGING_H_

#include <iostream>
#include <sstream>
#include <string>

namespace mobula {

class MobulaLogger {
 public:
  MobulaLogger(const std::string name, bool is_fatal)
      : name_(name), is_fatal_(is_fatal) {}
  template <typename T>
  MobulaLogger& operator<<(const T& msg) {
    buffer_ << msg;
    return *this;
  }
  template <typename T>
  MobulaLogger& operator<<(const T&& msg) {
    return operator<<(msg);
  }
  ~MobulaLogger() {
    std::cout << "[" << name_ << "] " << buffer_.str() << std::endl;
    if (is_fatal_) exit(-1);
  }

 private:
  std::string name_;
  bool is_fatal_;
  std::ostringstream buffer_;
};

#define LOG_INFO \
  mobula::MobulaLogger("INFO", false) << __FILE__ << ":" << __LINE__ << " "
#define LOG_WARNING \
  mobula::MobulaLogger("WARNING", false) << __FILE__ << ":" << __LINE__ << " "
#define LOG_FATAL \
  mobula::MobulaLogger("FATAL", true) << __FILE__ << ":" << __LINE__ << " "

#define LOG(KIND) LOG_##KIND

#define CHECK(x) \
  if (!(x)) LOG(FATAL)

#define CHECK_BINARY_OP(op, x, y)                       \
  if (!((x)op(y)))                                      \
  LOG(FATAL) << " Check Failed: " #x " " #op " " #y " " \
             << " " << x << " vs " << y

#define CHECK_EQ(x, y) CHECK_BINARY_OP(==, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(!=, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(<=, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(>=, x, y)
#define CHECK_LT(x, y) CHECK_BINARY_OP(<, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(>, x, y)

#ifdef MOBULA_LOG_DEBUG
#define DCHECK_EQ CHECK_EQ
#define DCHECK_NE CHECK_NE
#define DCHECK_LE CHECK_LE
#define DCHECK_GE CHECK_GE
#define DCHECK_LT CHECK_LT
#define DCHECK_GT CHECK_GT
#else
#define DCHECK_EQ
#define DCHECK_NE
#define DCHECK_LE
#define DCHECK_GE
#define DCHECK_LT
#define DCHECK_GT
#endif

}  // namespace mobula

#endif  // MOBULA_INC_LOGGING_H_
