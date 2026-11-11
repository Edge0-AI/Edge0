#include "model/asset_index.h"
#include "model/sha256.h"
#include "gguf.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
namespace edge0::model {
namespace {
Error invalid(const std::string &file, const std::string &why) {
    return {ErrorCategory::asset_invalid, file + ": " + why};
}
// This parser deliberately accepts only the catalog's JSON subset.
struct Json {
    std::string text;
    std::size_t pos = 0;
    void ws() {
        while (pos < text.size() &&
               (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
            ++pos;
    }
    void validate_utf8() const {
        std::size_t offset = 0;
        while (offset < text.size()) {
            const auto lead = static_cast<unsigned char>(text[offset++]);
            if (lead < 0x80)
                continue;
            unsigned continuation = 0;
            std::uint32_t code = 0, minimum = 0;
            if (lead >= 0xc2 && lead <= 0xdf) {
                continuation = 1;
                code = lead & 0x1f;
                minimum = 0x80;
            } else if (lead >= 0xe0 && lead <= 0xef) {
                continuation = 2;
                code = lead & 0x0f;
                minimum = 0x800;
            } else if (lead >= 0xf0 && lead <= 0xf4) {
                continuation = 3;
                code = lead & 0x07;
                minimum = 0x10000;
            } else {
                throw std::runtime_error("invalid UTF-8 leading byte");
            }
            if (continuation > text.size() - offset)
                throw std::runtime_error("truncated UTF-8 sequence");
            for (unsigned i = 0; i < continuation; ++i) {
                const auto byte = static_cast<unsigned char>(text[offset++]);
                if ((byte & 0xc0) != 0x80)
                    throw std::runtime_error("invalid UTF-8 continuation byte");
                code = (code << 6) | (byte & 0x3f);
            }
            if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
                throw std::runtime_error("invalid UTF-8 scalar value");
        }
    }
    bool take(char c) {
        ws();
        if (pos < text.size() && text[pos] == c) {
            ++pos;
            return true;
        }
        return false;
    }
    void need(char c) {
        if (!take(c))
            throw std::runtime_error("invalid JSON");
    }
    std::string str() {
        need('"');
        std::string s;
        while (pos < text.size()) {
            auto c = text[pos++];
            if (c == '"')
                return s;
            if (static_cast<unsigned char>(c) < 32)
                throw std::runtime_error("JSON control character");
            if (c == '\\') {
                if (pos == text.size())
                    break;
                c = text[pos++];
                switch (c) {
                case '"':
                case '\\':
                case '/':
                    break;
                case 'b':
                    c = '\b';
                    break;
                case 'f':
                    c = '\f';
                    break;
                case 'n':
                    c = '\n';
                    break;
                case 'r':
                    c = '\r';
                    break;
                case 't':
                    c = '\t';
                    break;
                case 'u': {
                    auto hex4 = [&]() {
                        unsigned value = 0;
                        for (int i = 0; i < 4; ++i) {
                            if (pos == text.size())
                                throw std::runtime_error("invalid unicode escape");
                            const char h = text[pos++];
                            value *= 16;
                            if (h >= '0' && h <= '9')
                                value += h - '0';
                            else if (h >= 'a' && h <= 'f')
                                value += h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F')
                                value += h - 'A' + 10;
                            else
                                throw std::runtime_error("invalid unicode escape");
                        }
                        return value;
                    };
                    unsigned code = hex4();
                    if (code >= 0xd800 && code <= 0xdbff) {
                        if (text.size() - pos < 2 || text[pos] != '\\' || text[pos + 1] != 'u')
                            throw std::runtime_error("unpaired high surrogate");
                        pos += 2;
                        const unsigned low = hex4();
                        if (low < 0xdc00 || low > 0xdfff)
                            throw std::runtime_error("invalid low surrogate");
                        code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                    } else if (code >= 0xdc00 && code <= 0xdfff) {
                        throw std::runtime_error("unpaired low surrogate");
                    }
                    if (code < 0x80)
                        s += static_cast<char>(code);
                    else if (code < 0x800) {
                        s += static_cast<char>(0xc0 | (code >> 6));
                        s += static_cast<char>(0x80 | (code & 63));
                    } else if (code < 0x10000) {
                        s += static_cast<char>(0xe0 | (code >> 12));
                        s += static_cast<char>(0x80 | ((code >> 6) & 63));
                        s += static_cast<char>(0x80 | (code & 63));
                    } else {
                        s += static_cast<char>(0xf0 | (code >> 18));
                        s += static_cast<char>(0x80 | ((code >> 12) & 63));
                        s += static_cast<char>(0x80 | ((code >> 6) & 63));
                        s += static_cast<char>(0x80 | (code & 63));
                    }
                    continue;
                }
                default:
                    throw std::runtime_error("invalid JSON escape");
                }
            }
            s += c;
        }
        throw std::runtime_error("unterminated JSON string");
    }
    std::variant<std::string, std::int64_t, bool, double, std::monostate> value(unsigned depth = 0) {
        ws();
        if (depth > 64 || pos == text.size())
            throw std::runtime_error("invalid JSON value");
        if (text[pos] == '"')
            return str();
        if (text[pos] == '{') {
            object(depth + 1);
            return false;
        }
        if (take('[')) {
            if (!take(']')) {
                do { value(depth + 1); } while (take(','));
                need(']');
            }
            return std::monostate{};
        }
        if (text.compare(pos, 4, "null") == 0) { pos += 4; return std::monostate{}; }
        for (const auto *word : {"true", "false"})
            if (text.compare(pos, std::strlen(word), word) == 0) {
                pos += std::strlen(word);
                return std::string(word) == "true";
            }
        auto start = pos;
        if (text[pos] == '-')
            ++pos;
        if (pos == text.size() || !std::isdigit(static_cast<unsigned char>(text[pos])))
            throw std::runtime_error("expected integer");
        if (text[pos] == '0')
            ++pos;
        else
            while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])))
                ++pos;
        bool floating = false;
        if (pos < text.size() && text[pos] == '.') {
            floating = true; ++pos;
            const auto digits = pos;
            while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) ++pos;
            if (pos == digits) throw std::runtime_error("invalid fraction");
        }
        if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
            floating = true; ++pos;
            if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) ++pos;
            const auto digits = pos;
            while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) ++pos;
            if (pos == digits) throw std::runtime_error("invalid exponent");
        }
        const auto number = text.substr(start, pos - start);
        if (floating) return std::stod(number);
        return std::stoll(number);
    }
    std::map<std::string, std::variant<std::string, std::int64_t, bool, double, std::monostate>> object(unsigned depth = 0) {
        need('{');
        std::map<std::string, std::variant<std::string, std::int64_t, bool, double, std::monostate>> obj;
        if (take('}'))
            return obj;
        do {
            auto key = str();
            need(':');
            if (!obj.emplace(key, value(depth)).second)
                throw std::runtime_error("duplicate JSON key");
        } while (take(','));
        need('}');
        return obj;
    }
};
std::vector<std::string> split(const std::string &line) {
    std::vector<std::string> out;
    std::size_t start = 0;
    for (;;) {
        auto end = line.find('\t', start);
        out.push_back(line.substr(start, end - start));
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    return out;
}
std::string shape(const std::array<std::int64_t, 4> &ne) {
    std::ostringstream out;
    for (auto n : ne)
        out << n << ',';
    return out.str();
}
template <class T> MetadataValue array(const gguf_context *g, std::int64_t key) {
    auto n = gguf_get_arr_n(g, key);
    std::vector<T> out(n);
    if (n)
        std::memcpy(out.data(), gguf_get_arr_data(g, key), n * sizeof(T));
    return out;
}
MetadataValue metadata_value(const gguf_context *g, std::int64_t key) {
    auto type = gguf_get_kv_type(g, key);
    if (type == GGUF_TYPE_ARRAY) {
        switch (gguf_get_arr_type(g, key)) {
#define ARRAY(tag, cpp)                                                                                      \
    case GGUF_TYPE_##tag:                                                                                    \
        return array<cpp>(g, key);
            ARRAY(UINT8, std::uint8_t)
            ARRAY(INT8, std::int8_t)
            ARRAY(UINT16, std::uint16_t)
            ARRAY(INT16, std::int16_t)
        ARRAY(UINT32, std::uint32_t)
        ARRAY(INT32, std::int32_t) ARRAY(UINT64, std::uint64_t) ARRAY(INT64, std::int64_t)
            ARRAY(FLOAT32, float) ARRAY(FLOAT64, double)
#undef ARRAY
                case GGUF_TYPE_BOOL: {
            std::vector<bool> v;
            const auto *p = static_cast<const std::int8_t *>(gguf_get_arr_data(g, key));
            for (std::size_t i = 0; i < gguf_get_arr_n(g, key); ++i)
                v.push_back(p[i] != 0);
            return v;
        }
        case GGUF_TYPE_STRING: {
            std::vector<std::string> v;
            for (std::size_t i = 0; i < gguf_get_arr_n(g, key); ++i)
                v.emplace_back(gguf_get_arr_str(g, key, i));
            return v;
        }
        default:
            throw std::runtime_error("unsupported metadata array");
        }
    }
    switch (type) {
#define SCALAR(tag, fn)                                                                                      \
    case GGUF_TYPE_##tag:                                                                                    \
        return gguf_get_val_##fn(g, key);
        SCALAR(UINT8, u8)
        SCALAR(INT8, i8)
        SCALAR(UINT16, u16)
        SCALAR(INT16, i16)
    SCALAR(UINT32, u32)
    SCALAR(INT32, i32) SCALAR(UINT64, u64) SCALAR(INT64, i64) SCALAR(FLOAT32, f32) SCALAR(FLOAT64, f64)
        SCALAR(BOOL, bool)
#undef SCALAR
            case GGUF_TYPE_STRING:
        return std::string(gguf_get_val_str(g, key));
        default : throw std::runtime_error("unsupported metadata type");
    }
}
// GGML normalizes tensor names through a C string while parsing. Verify only
// their original identity; shape, type, layout and offset validation remain in
// GGML. Its no-tensor parser exposes the exact (unpadded) end of the KV section.
std::optional<Error> check_tensor_names(const gguf_context *parsed, const AssetFile &file) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_empty(), gguf_free);
    gguf_set_kv(metadata.get(), parsed);
    std::vector<char> encoded(gguf_get_meta_size(metadata.get()));
    gguf_get_meta_data(metadata.get(), encoded.data());
    metadata.reset(gguf_init_from_buffer(encoded.data(), encoded.size(), {true, nullptr}));
    if (!metadata)
        return invalid(file.filename, "cannot locate GGUF tensor names");
    std::uint64_t position = gguf_get_data_offset(metadata.get());
    const std::uint64_t end = gguf_get_data_offset(parsed);
    // A deleter type rather than decltype(&std::fclose): GCC warns that the template argument drops fclose's attributes.
    struct CloseFile {
        void operator()(FILE* handle) const { std::fclose(handle); }
    };
    std::unique_ptr<FILE, CloseFile> input(std::fopen(file.path.c_str(), "rb"));
    if (!input)
        return invalid(file.filename, "cannot read GGUF tensor names");
    // Prevent stdio read-ahead across the header/payload boundary.
    if (std::setvbuf(input.get(), nullptr, _IONBF, 0) != 0)
        return invalid(file.filename, "cannot configure GGUF header reader");
    auto seek = [&](std::uint64_t target) {
        if (target > end || target > static_cast<std::uint64_t>(std::numeric_limits<long>::max()))
            return false;
        if (std::fseek(input.get(), static_cast<long>(target), SEEK_SET) != 0)
            return false;
        position = target;
        return true;
    };
    auto read = [&](void *data, std::size_t bytes) {
        if (position > end || bytes > end - position)
            return false;
        if (std::fread(data, 1, bytes, input.get()) != bytes)
            return false;
        position += bytes;
        return true;
    };
    if (!seek(position))
        return invalid(file.filename, "invalid GGUF tensor table boundary");
    for (std::int64_t i = 0; i < gguf_get_n_tensors(parsed); ++i) {
        std::uint64_t length = 0;
        if (!read(&length, sizeof(length)) || length > end - position)
            return invalid(file.filename, "invalid GGUF tensor name boundary");
        std::string name(static_cast<std::size_t>(length), '\0');
        if (!read(name.data(), name.size()))
            return invalid(file.filename, "truncated GGUF tensor name");
        if (name.find('\0') != std::string::npos || name != gguf_get_tensor_name(parsed, i))
            return invalid(file.filename, "GGUF tensor name loses identity through the C API");
        std::uint32_t dimensions = 0;
        if (!read(&dimensions, sizeof(dimensions)))
            return invalid(file.filename, "truncated tensor record");
        // Skip original dimensions (including trailing ones), type and relative
        // offset without interpreting them or imposing canonical serialization.
        const std::uint64_t rest =
            std::uint64_t(dimensions) * sizeof(std::uint64_t) + sizeof(std::uint32_t) + sizeof(std::uint64_t);
        if (position > end || rest > end - position || !seek(position + rest))
            return invalid(file.filename, "invalid GGUF tensor record boundary");
    }
    return std::nullopt;
}
} // namespace
Result<Catalog> open_catalog(const std::string &directory, Integrity integrity,
                             const std::vector<std::string> *architectures) {
    if (directory.find('\0') != std::string::npos)
        return invalid("model.json", "directory contains NUL");
    const auto root = std::filesystem::path(directory);
    Catalog out;
    out.integrity = integrity;
    try {
        std::ifstream in(root / "model.json");
        if (!in)
            return invalid("model.json", "cannot read");
        Json parser{std::string(std::istreambuf_iterator<char>(in), {})};
        parser.validate_utf8();
        auto obj = parser.object();
        parser.ws();
        if (parser.pos != parser.text.size())
            return invalid("model.json", "trailing JSON");
        auto version = obj.find("format_version"), arch = obj.find("architecture");
        if (version == obj.end() && arch == obj.end())
            return Error{ErrorCategory::unsupported,
                         "model.json is not an Edge0 asset manifest (it has neither format_version nor architecture); "
                         "models that run on an external engine (such as Edge0 145B) are served with "
                         "`edge0-infer serve` or `edge0-infer chat`"};
        if (version == obj.end() || !std::holds_alternative<std::int64_t>(version->second) ||
            std::get<std::int64_t>(version->second) != 1)
            return invalid("model.json", "unsupported format_version");
        if (arch == obj.end() || !std::holds_alternative<std::string>(arch->second) ||
            std::get<std::string>(arch->second).empty())
            return invalid("model.json", "architecture must be nonempty string");
        out.architecture = std::get<std::string>(arch->second);
    } catch (const std::exception &e) {
        return invalid("model.json", e.what());
    }
    if (architectures &&
        std::find(architectures->begin(), architectures->end(), out.architecture) == architectures->end()) {
        // Decided before assets.tsv or any weight file is read: the directory belongs to an
        // implementation this library does not have, such as a model run by an external engine.
        std::string known;
        for (const auto &name : *architectures)
            known += (known.empty() ? "" : ", ") + name;
        return Error{ErrorCategory::unsupported,
                     "architecture " + out.architecture + " is not compiled into this library (compiled: " + known +
                         "); models that run on an external engine (such as Edge0 145B) are served with "
                         "`edge0-infer serve` or `edge0-infer chat`"};
    }
    std::ifstream in(root / "assets.tsv");
    std::string line;
    if (!std::getline(in, line) || line != "role\tbytes\tsha256\tfilename")
        return invalid("assets.tsv", "missing or invalid header");
    while (std::getline(in, line)) {
        auto cols = split(line);
        if (cols.size() != 4)
            return invalid("assets.tsv", "expected four columns");
        const auto &name = cols[3];
        if (name.find('\0') != std::string::npos || name.empty() || name == "." ||
            name.find("..") != std::string::npos || name.find_first_of("/\\") != std::string::npos ||
            std::filesystem::path(name).is_absolute())
            return invalid("assets.tsv", name + ": invalid filename");
        if (cols[0].empty() || out.files.count(cols[0]))
            return invalid("assets.tsv", name + ": duplicate or empty role " + cols[0]);
        if (cols[2].size() != 64 || cols[2].find_first_not_of("0123456789abcdef") != std::string::npos)
            return invalid("assets.tsv", name + ": invalid sha256");
        std::uint64_t bytes;
        try {
            if (cols[1].empty() || cols[1].find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("bytes");
            bytes = std::stoull(cols[1]);
        } catch (const std::exception &) {
            return invalid("assets.tsv", name + ": invalid bytes");
        }
        std::error_code ec;
        auto path = root / name;
        if (!std::filesystem::is_regular_file(path, ec) || ec)
            return invalid(name, "not a regular file");
        auto size = std::filesystem::file_size(path, ec);
        if (ec || size != bytes)
            return invalid(name, "size mismatch");
        AssetFile file{cols[0], name, path.string(), bytes, cols[2], false};
        if (integrity == Integrity::digest) {
            auto hash = sha256_file(file.path);
            if (!hash)
                return hash.error();
            if (hash.value() != file.sha256)
                return invalid(name, "sha256 mismatch");
            file.digest_verified = true;
        }
        out.files.emplace(file.role, std::move(file));
    }
    if (!in.eof())
        return invalid("assets.tsv", "read failed");
    return out;
}
Result<TensorIndex> index_tensors(const AssetFile &file) {
    std::error_code ec;
    auto size = std::filesystem::file_size(file.path, ec);
    if (ec)
        return invalid(file.filename, "cannot stat");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(
        gguf_init_from_file(file.path.c_str(), {true, nullptr}), gguf_free);
    if (!g)
        return invalid(file.filename, "GGUF header rejected by parser");
    // GGML's C string getters cannot express embedded NUL bytes. Round-trip
    // through its own setters and serializer to reject any lossy conversion.
    // No payload is serialized or read by this check.
    std::vector<char> original(gguf_get_meta_size(g.get()));
    gguf_get_meta_data(g.get(), original.data());
    std::unique_ptr<gguf_context, decltype(&gguf_free)> copy(
        gguf_init_from_buffer(original.data(), original.size(), {true, nullptr}), gguf_free);
    if (!copy)
        return invalid(file.filename, "GGUF metadata cannot be represented");
    gguf_set_kv(copy.get(), g.get());
    std::vector<char> roundtrip(gguf_get_meta_size(copy.get()));
    gguf_get_meta_data(copy.get(), roundtrip.data());
    if (original != roundtrip)
        return invalid(file.filename, "GGUF metadata contains strings not representable by the C API");
    copy.reset();
    original.clear();
    roundtrip.clear();
    if (auto error = check_tensor_names(g.get(), file))
        return *error;
    TensorIndex index;
    index.file = file;
    try {
        for (std::int64_t i = 0; i < gguf_get_n_kv(g.get()); ++i)
            index.metadata.emplace(gguf_get_key(g.get(), i), metadata_value(g.get(), i));
        for (std::int64_t i = 0; i < gguf_get_n_tensors(g.get()); ++i) {
            Tensor t;
            t.name = gguf_get_tensor_name(g.get(), i);
            t.role = file.role;
            t.path = file.path;
            auto type = gguf_get_tensor_type(g.get(), i);
            t.type = ggml_type_name(type);
            std::copy_n(gguf_get_tensor_ne(g.get(), i), 4, t.ne.begin());
            t.block_size = static_cast<std::uint64_t>(ggml_blck_size(type));
            t.nb[0] = ggml_type_size(type);
            t.nb[1] = t.nb[0] * (static_cast<std::uint64_t>(t.ne[0]) / t.block_size);
            for (int d = 2; d < 4; ++d)
                t.nb[d] = t.nb[d - 1] * static_cast<std::uint64_t>(t.ne[d - 1]);
            auto start = gguf_get_data_offset(g.get()), offset = gguf_get_tensor_offset(g.get(), i);
            t.bytes = gguf_get_tensor_size(g.get(), i);
            if (start > size || offset > size - start || t.bytes > size - start - offset)
                return invalid(file.filename, t.name + ": tensor data exceeds file end");
            t.offset = start + offset;
            index.tensors.emplace(t.name, std::move(t));
        }
    } catch (const std::exception &e) {
        return invalid(file.filename, e.what());
    }
    return index;
}
Result<const Tensor *> find_tensor(const TensorIndex &index, const TensorRequirement &req) {
    auto found = index.tensors.find(req.name);
    auto fail = [&](const std::string &why) {
        return Error{ErrorCategory::incompatible, req.role + "/" + req.name + ": " + why};
    };
    if (found == index.tensors.end())
        return fail("expected tensor; actual missing");
    const auto &t = found->second;
    if (t.type != req.type)
        return fail("expected type " + req.type + "; actual " + t.type);
    if (t.ne != req.shape)
        return fail("expected shape " + shape(req.shape) + "; actual " + shape(t.ne));
    return &t;
}
Result<Slice> locate_slice(const Tensor &t, std::uint64_t first, std::uint64_t count) {
    auto fail = [&]() {
        return Error{ErrorCategory::incompatible, t.name + ": invalid or noncontiguous slice"};
    };
    int outer = 3;
    while (outer > 0 && t.ne[outer] == 1)
        --outer;
    for (auto n : t.ne)
        if (n <= 0)
            return fail();
    auto extent = static_cast<std::uint64_t>(t.ne[outer]);
    if (!count || first >= extent || count > extent - first || !t.block_size)
        return fail();
    if (!t.nb[0] || static_cast<std::uint64_t>(t.ne[0]) % t.block_size ||
        static_cast<std::uint64_t>(t.ne[0]) / t.block_size >
            std::numeric_limits<std::uint64_t>::max() / t.nb[0] ||
        t.nb[1] != t.nb[0] * (static_cast<std::uint64_t>(t.ne[0]) / t.block_size))
        return fail();
    for (int d = 2; d <= outer; ++d)
        if (!t.nb[d - 1] ||
            static_cast<std::uint64_t>(t.ne[d - 1]) >
                std::numeric_limits<std::uint64_t>::max() / t.nb[d - 1] ||
            t.nb[d] != t.nb[d - 1] * static_cast<std::uint64_t>(t.ne[d - 1]))
            return fail();
    std::uint64_t stride = outer ? t.nb[outer] : t.nb[0];
    if (!outer) {
        if (first % t.block_size || count % t.block_size)
            return fail();
        first /= t.block_size;
        count /= t.block_size;
    }
    if (!stride || first > (std::numeric_limits<std::uint64_t>::max() - t.offset) / stride ||
        count > t.bytes / stride)
        return fail();
    auto off = first * stride, len = count * stride;
    if (off > t.bytes || len > t.bytes - off)
        return fail();
    return Slice{t.path, t.type, t.offset + off, len};
}
} // namespace edge0::model

namespace edge0::model {
Result<GenerationDefaults> read_generation_defaults(const std::string& directory) {
    const auto path = std::filesystem::path(directory) / "generation_config.json";
    try {
        if (!std::filesystem::exists(path)) return GenerationDefaults{};
        std::ifstream file(path, std::ios::binary);
        if (!file) return invalid("generation_config.json", "cannot open");
        Json parser{std::string(std::istreambuf_iterator<char>(file), {})};
        parser.validate_utf8();
        const auto values = parser.object();
        parser.ws();
        if (parser.pos != parser.text.size()) throw std::runtime_error("trailing JSON data");
        GenerationDefaults result;
        for (const auto& field : values) {
            const auto& name = field.first;
            if (name != "temperature" && name != "top_k" && name != "top_p" && name != "repetition_penalty") continue;
            if (name == "top_k") {
                const auto* value = std::get_if<std::int64_t>(&field.second);
                if (!value || *value < 0) throw std::runtime_error("invalid top_k");
                result.top_k = *value;
                continue;
            }
            double value;
            if (const auto* integer = std::get_if<std::int64_t>(&field.second)) value = static_cast<double>(*integer);
            else if (const auto* number = std::get_if<double>(&field.second)) value = *number;
            else throw std::runtime_error("invalid " + name + " type");
            if (!std::isfinite(value) || (name == "temperature" ? value < 0 : value <= 0) ||
                (name == "top_p" && value > 1)) throw std::runtime_error("invalid " + name + " range");
            if (name == "temperature") result.temperature = value;
            else if (name == "top_p") result.top_p = value;
            else result.repetition_penalty = value;
        }
        return result;
    } catch (const std::exception& error) { return invalid("generation_config.json", error.what()); }
}
}
