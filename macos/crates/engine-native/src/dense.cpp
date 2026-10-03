#include "dense.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstring>

namespace e0n {
namespace {

bool read_range(const ModelFiles& files, const TensorInfo& t, std::vector<uint8_t>& out,
                std::string& err) {
  if (t.end < t.begin) { err = "tensor range inverted: " + t.shard; return false; }
  out.resize(static_cast<size_t>(t.end - t.begin));
  int fd = ::open(files.shard_path(t.shard).c_str(), O_RDONLY);
  if (fd < 0) { err = "open failed: " + files.shard_path(t.shard); return false; }
  size_t done = 0;
  while (done < out.size()) {
    ssize_t got = ::pread(fd, out.data() + done, out.size() - done,
                          static_cast<off_t>(t.begin + done));
    if (got <= 0) {
      ::close(fd);
      err = "pread failed: " + t.shard;
      return false;
    }
    done += static_cast<size_t>(got);
  }
  ::close(fd);  // one-shot resident read; page-cache open is fine (not the hot-stack F_NOCACHE path)
  return true;
}

}  // namespace

float f16_bits_to_f(uint16_t b) {
  _Float16 h;
  __builtin_memcpy(&h, &b, 2);
  return static_cast<float>(h);
}

bool read_dense_f32(const ModelFiles& files, const std::string& tensor, DenseF32& out,
                    std::string& err) {
  const TensorInfo* t = files.find(tensor);
  if (!t) { err = "missing tensor: " + tensor; return false; }
  if (t->shape.empty()) { err = "dense rejects scalar tensors: " + tensor; return false; }
  std::vector<uint8_t> bytes;
  if (!read_range(files, *t, bytes, err)) return false;
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < t->shape.size(); i++) rows *= t->shape[i];
  int64_t cols = t->shape.back();
  size_t n = static_cast<size_t>(rows * cols);
  out.rows = rows;
  out.cols = cols;
  out.v.resize(n);
  if (t->dtype == "F32") {
    if (bytes.size() != n * 4) { err = "F32 byte size mismatch: " + tensor; return false; }
    std::memcpy(out.v.data(), bytes.data(), bytes.size());
  } else if (t->dtype == "BF16") {
    if (bytes.size() != n * 2) { err = "BF16 byte size mismatch: " + tensor; return false; }
    const uint16_t* p = reinterpret_cast<const uint16_t*>(bytes.data());
    for (size_t i = 0; i < n; i++) out.v[i] = bf16_bits_to_f(p[i]);
  } else if (t->dtype == "F16") {
    if (bytes.size() != n * 2) { err = "F16 byte size mismatch: " + tensor; return false; }
    const uint16_t* p = reinterpret_cast<const uint16_t*>(bytes.data());
    for (size_t i = 0; i < n; i++) out.v[i] = f16_bits_to_f(p[i]);
  } else if (t->dtype == "I32" || t->dtype == "I64") {
    size_t esz = t->dtype == "I32" ? 4 : 8;
    if (bytes.size() != n * esz) { err = "integer byte size mismatch: " + tensor; return false; }
    for (size_t i = 0; i < n; i++) {
      if (esz == 4) { int32_t v; std::memcpy(&v, bytes.data() + 4 * i, 4); out.v[i] = (float)v; }
      else { int64_t v; std::memcpy(&v, bytes.data() + 8 * i, 8); out.v[i] = (float)v; }
    }
  } else {
    err = "unsupported dense dtype: " + t->dtype + " (" + tensor + ")";
    return false;
  }
  return true;
}

void dequant_row(const uint32_t* packed, const uint16_t* scales_row,
                 const uint16_t* biases_row, int64_t cols_packed, int bits,
                 int64_t group_size, std::vector<float>& dst_row) {
  const int per_word = 32 / bits;          // int4→8, int8→4
  const uint32_t mask = (1u << bits) - 1u;  // 0xF / 0xFF
  const int64_t cols = cols_packed * per_word;
  dst_row.resize(static_cast<size_t>(cols));
  for (int64_t c = 0; c < cols; c++) {
    int64_t word = c / per_word;
    int shift = static_cast<int>((c % per_word) * bits);
    uint32_t q = (packed[word] >> shift) & mask;  // little-endian grouping
    int64_t g = c / group_size;
    double s = bf16_bits_to_f(scales_row[g]);
    double b = bf16_bits_to_f(biases_row[g]);
    dst_row[c] = static_cast<float>(static_cast<double>(q) * s + b);  // q*scale + bias
  }
}

bool read_dense_dequant(const ModelFiles& files, const std::string& tensor, int64_t bits,
                        int64_t group_size, DenseF32& out, std::string& err) {
  if (bits != 4 && bits != 8) { err = "dense dequant supports only 4/8 bit"; return false; }
  const int per_word = static_cast<int>(32 / bits);
  // tensor = "...gate.weight" → module path = strip last segment; scales/biases hang there.
  std::string mod = tensor;
  auto dot = mod.rfind('.');
  if (dot != std::string::npos && mod.substr(dot + 1) == "weight") mod = mod.substr(0, dot);
  const TensorInfo *w = files.find(tensor), *s = files.find(mod + ".scales"),
                   *b = files.find(mod + ".biases");
  if (!w || !s || !b) { err = "missing weight/scales/biases: " + tensor; return false; }
  if (w->dtype != "U32" || s->dtype != "BF16" || b->dtype != "BF16") {
    err = "quantized triple dtype mismatch: " + tensor;
    return false;
  }
  std::vector<uint8_t> wb, sb, bb;
  if (!read_range(files, *w, wb, err) || !read_range(files, *s, sb, err) ||
      !read_range(files, *b, bb, err))
    return false;
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < w->shape.size(); i++) rows *= w->shape[i];
  int64_t cp = w->shape.back();                 // packed last dim
  int64_t cols = cp * per_word;                 // logical K
  int64_t groups = cols / group_size;
  if (s->shape.back() != groups || static_cast<int64_t>(sb.size()) != rows * groups * 2 ||
      static_cast<int64_t>(bb.size()) != rows * groups * 2) {
    err = "scales/biases shape mismatch: " + tensor;
    return false;
  }
  out.rows = rows;
  out.cols = cols;
  out.v.resize(static_cast<size_t>(rows * cols));
  const uint32_t* wp = reinterpret_cast<const uint32_t*>(wb.data());
  const uint16_t* sp = reinterpret_cast<const uint16_t*>(sb.data());
  const uint16_t* bp = reinterpret_cast<const uint16_t*>(bb.data());
  for (int64_t r = 0; r < rows; r++) {
    std::vector<float> row;
    dequant_row(wp + r * cp, sp + r * groups, bp + r * groups, cp, static_cast<int>(bits),
                group_size, row);
    std::memcpy(out.v.data() + r * cols, row.data(), row.size() * sizeof(float));
  }
  return true;
}

}  // namespace e0n
