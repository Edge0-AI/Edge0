#pragma once
#include <string>
namespace edge0::text {
class Decoder {
public:
  std::string decode_increment(const std::string &bytes);
  std::string finish();

private:
  std::string drain(bool final);
  std::string pending_;
};
} // namespace edge0::text
