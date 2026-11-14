// metal/src/gguf_lite.h - a minimal GGUF v3 reader for the Metal test programs: tensor names, shapes, types and
// file offsets, the file mmapped read-only (no copy; wrapped as one MTLBuffer by the caller).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gguf {

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;   // ggml order: shape[0] is the contiguous (row) dimension
    int type = -1;
    uint64_t offset = 0;           // from the start of the FILE (data start + tensor offset)
    uint64_t bytes = 0;
};

struct File {
    std::string path;
    void* map = nullptr;           // page-aligned mapping of the whole file
    size_t map_bytes = 0;          // rounded up to the page size
    size_t file_bytes = 0;
    std::map<std::string, Tensor> tensors;
    std::map<std::string, std::string> meta_str;
    std::map<std::string, double> meta_num;
    bool open(const std::string& path, std::string& err);
    const Tensor* find(const std::string& name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }
    ~File();
};

}  // namespace gguf
