#include "strata/core/latent_dense.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {

#if defined(_WIN32)
// Windows: CreateFileA + CreateFileMappingA + MapViewOfFile.  Same contract as the POSIX path:
// strict size check, read-only mapping, unmap on close.  The mapping handle keeps the section
// alive after the file handle closes.
static void win_err(std::string& err, const std::string& path, const char* what) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), " (win32 error %lu)", (unsigned long) GetLastError());
    err = path + ": " + what + buf;
}
#endif

LatentDense::~LatentDense() { close(); }

void LatentDense::close() {
#if defined(_WIN32)
    if (base_) UnmapViewOfFile(base_);
    if (map_) CloseHandle(map_);
#else
    if (base_) munmap(base_, size_);
#endif
    base_ = nullptr; map_ = nullptr;
    size_ = 0; bytes_ = 0;
}

bool LatentDense::open(const std::string& path, int64_t n_layers, int64_t n_groups, int64_t n_embd,
                       int64_t latent_dim, std::string& err) {
    close();
    n_layers_ = n_layers; n_groups_ = n_groups; n_embd_ = n_embd; latent_dim_ = latent_dim;
    // Q4DU: STRATA_LATENT_DENSE_FMT=4 selects the Q4_0 pack (18 B/block), 8 or unset the
    // historical Q8_0 one (34 B/block).  The file's size must then match the derived layout -
    // a mismatched file/env pair fails the open with both candidate sizes spelled out.
    fmt_ = 8;
    if (const char* f = ::getenv("STRATA_LATENT_DENSE_FMT")) {
        if (f[0] == '4' && f[1] == 0) fmt_ = 4;
        else if (f[0] == '8' && f[1] == 0) fmt_ = 8;
        else { err = path + ": STRATA_LATENT_DENSE_FMT must be 4 or 8 (got '" + std::string(f) + "')"; return false; }
    }
    const size_t db = d_bytes(), ub = u_bytes();
    const uint64_t want = (uint64_t) n_layers * (db + ub);
#if defined(_WIN32)
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) { win_err(err, path, "CreateFile"); return false; }
    LARGE_INTEGER fsz;
    if (!GetFileSizeEx(fh, &fsz)) { win_err(err, path, "GetFileSizeEx"); CloseHandle(fh); return false; }
    if ((uint64_t) fsz.QuadPart != want) {
        const uint64_t alt = (uint64_t) n_layers *
            ((size_t) n_groups_ * (size_t) latent_dim_ * (size_t) (n_embd_ / 32) * (fmt_ == 4 ? 34 : 18) +
             (size_t) n_groups_ * (size_t) n_embd_ * (size_t) (latent_dim_ / 32) * (fmt_ == 4 ? 34 : 18));
        err = path + ": size " + std::to_string((long long) fsz.QuadPart) + " != " + std::to_string(want) +
              " (fmt " + std::to_string(fmt_) + " wants " + std::to_string(want) +
              "; the other fmt wants " + std::to_string((long long) alt) + ")";
        CloseHandle(fh); return false;
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) { win_err(err, path, "CreateFileMapping"); CloseHandle(fh); return false; }
    CloseHandle(fh);   // the mapping object holds the section
    void* p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!p) { win_err(err, path, "MapViewOfFile"); CloseHandle(mh); return false; }
    map_ = mh;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = path + ": " + std::strerror(errno); return false; }
    struct stat st;
    if (fstat(fd, &st) != 0) { err = path + ": fstat: " + std::strerror(errno); ::close(fd); return false; }
    if ((uint64_t) st.st_size != want) {   // strict: the env's format and the file's size disagree.
        // NO silent fallback - a misread layout would produce garbage answers.  Name both
        // candidate sizes so the operator sees exactly which format the file actually is.
        const uint64_t alt = (uint64_t) n_layers *
            ((size_t) n_groups_ * (size_t) latent_dim_ * (size_t) (n_embd_ / 32) * (fmt_ == 4 ? 34 : 18) +
             (size_t) n_groups_ * (size_t) n_embd_ * (size_t) (latent_dim_ / 32) * (fmt_ == 4 ? 34 : 18));
        err = path + ": size " + std::to_string((long long) st.st_size) + " != " + std::to_string(want) +
              " (fmt " + std::to_string(fmt_) + " wants " + std::to_string(want) +
              "; the other fmt wants " + std::to_string((long long) alt) + ")";
        ::close(fd); return false;
    }
    void* p = mmap(nullptr, (size_t) want, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) { err = path + ": mmap: " + std::strerror(errno); return false; }
#endif
    base_ = (uint8_t*) p; size_ = (size_t) want; bytes_ = want;
    return true;
}

const uint8_t* LatentDense::d(int64_t layer) const {
    if (!base_ || layer < 0 || layer >= n_layers_) return nullptr;
    return base_ + (size_t) layer * (d_bytes() + u_bytes());
}

const uint8_t* LatentDense::u(int64_t layer) const {
    if (!base_ || layer < 0 || layer >= n_layers_) return nullptr;
    return d(layer) + d_bytes();
}

LatentA::~LatentA() { close(); }
void LatentA::close() {
#if defined(_WIN32)
    if (base_) UnmapViewOfFile(base_);
    if (map_) CloseHandle(map_);
#else
    if (base_) munmap(base_, size_);
#endif
    base_ = nullptr; map_ = nullptr; size_ = 0;
}
bool LatentA::open(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t blob_bytes, std::string& err) {
    close();
    n_layers_ = n_layers; n_expert_ = n_expert; blob_bytes_ = blob_bytes;
    const uint64_t want = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) blob_bytes;
#if defined(_WIN32)
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) { win_err(err, path, "CreateFile"); return false; }
    LARGE_INTEGER fsz;
    if (!GetFileSizeEx(fh, &fsz)) { win_err(err, path, "GetFileSizeEx"); CloseHandle(fh); return false; }
    if ((uint64_t) fsz.QuadPart != want) {
        err = path + ": size " + std::to_string((long long) fsz.QuadPart) + " != " + std::to_string(want);
        CloseHandle(fh); return false;
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) { win_err(err, path, "CreateFileMapping"); CloseHandle(fh); return false; }
    CloseHandle(fh);
    void* p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!p) { win_err(err, path, "MapViewOfFile"); CloseHandle(mh); return false; }
    map_ = mh;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = path + ": " + std::strerror(errno); return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || (uint64_t) st.st_size != want) {
        err = path + ": size " + std::to_string((long long) st.st_size) + " != " + std::to_string(want);
        ::close(fd); return false;
    }
    void* p = mmap(nullptr, (size_t) want, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) { err = path + ": mmap: " + std::strerror(errno); return false; }
#endif
    base_ = (uint8_t*) p; size_ = (size_t) want;
    return true;
}
const uint8_t* LatentA::blob(int64_t layer, int64_t expert) const {
    if (!base_ || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return nullptr;
    return base_ + ((size_t) layer * (size_t) n_expert_ + (size_t) expert) * (size_t) blob_bytes_;
}

}  // namespace strata::core
