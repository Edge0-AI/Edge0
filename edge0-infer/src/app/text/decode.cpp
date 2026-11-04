#include "app/text/decode.h"
#include "app/text/bpe.h"
namespace edge0::text {
namespace {
// The prefix length consumed on an error matches Python's maximal-subpart rule.
size_t sequence(const std::string &s, size_t i, bool &valid, bool &incomplete) {
  const auto c = static_cast<unsigned char>(s[i]);
  valid = false;
  incomplete = false;
  if (c < 0x80) {
    valid = true;
    return 1;
  }
  const size_t n = c >= 0xc2 && c <= 0xdf   ? 2
                   : c >= 0xe0 && c <= 0xef ? 3
                   : c >= 0xf0 && c <= 0xf4 ? 4
                                            : 1;
  if (n == 1)
    return 1;
  for (size_t k = 1; k < n; ++k) {
    if (i + k == s.size()) {
      incomplete = true;
      return k;
    }
    const auto b = static_cast<unsigned char>(s[i + k]);
    // CPython buffers ED A0..BF until a third byte or final=True, even
    // though that prefix cannot become a Unicode scalar value.
    if (k == 1 && c == 0xed && b >= 0xa0 && b <= 0xbf) {
      incomplete = i + 2 == s.size();
      return 1;
    }
    if (b < 0x80 || b > 0xbf ||
        (k == 1 && ((c == 0xe0 && b < 0xa0) || (c == 0xf0 && b < 0x90) ||
                    (c == 0xf4 && b >= 0x90))))
      return k;
  }
  valid = true;
  return n;
}
} // namespace
bool valid_utf8(const std::string &s) {
  for (size_t i = 0; i < s.size();) {
    bool good, tail;
    size_t n = sequence(s, i, good, tail);
    if (!good)
      return false;
    i += n;
  }
  return true;
}
std::string Decoder::decode_increment(const std::string &bytes) {
  pending_ += bytes;
  return drain(false);
}
std::string Decoder::finish() { return drain(true); }
std::string Decoder::drain(bool final) {
  std::string out;
  size_t i = 0;
  while (i < pending_.size()) {
    bool good, tail;
    size_t n = sequence(pending_, i, good, tail);
    if (tail && !final)
      break;
    if (good)
      out.append(pending_, i, n);
    else
      out += "\xef\xbf\xbd";
    i += n;
  }
  pending_.erase(0, i);
  return out;
}
} // namespace edge0::text
