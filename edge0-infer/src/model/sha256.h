#pragma once
#include "edge0/result.h"
#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
namespace edge0::model {
class Sha256 {
  public:
    void update(const void *data, std::size_t size);
    void update(std::string_view value) {
        update(value.data(), value.size());
    }
    std::string finish() const;

  private:
    void block(const std::uint8_t *data);
    std::array<std::uint32_t, 8> state_{
        {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t bytes_ = 0;
    std::size_t used_ = 0;
};
Result<std::string> sha256_file(const std::string &path);
} // namespace edge0::model
