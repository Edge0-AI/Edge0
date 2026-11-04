#include "tests/support/model_fixture.h"
#include <cstring>
extern "C" void model_read_probe_start(const char *, std::uint64_t);
extern "C" bool model_read_probe_stop();
int main() {
    using namespace fixture;
    using edge0::ErrorCategory;
    Temp t;
    Metadata meta;
    meta["u32"] = std::uint32_t(42);
    meta["float"] = 1.5f;
    meta["bool"] = true;
    meta["string"] = std::string("hello");
    meta["u32s"] = std::vector<std::uint32_t>{2, 3};
    meta["bools"] = std::vector<bool>{false, true};
    meta["strings"] = std::vector<std::string>{"one", "two"};
    meta["u8"] = std::uint8_t(8);
    meta["i8"] = std::int8_t(-8);
    meta["u16"] = std::uint16_t(16);
    meta["i16"] = std::int16_t(-16);
    meta["i32"] = std::int32_t(-32);
    meta["u64"] = std::uint64_t(64);
    meta["i64"] = std::int64_t(-64);
    meta["double"] = 2.5;
    meta["u8s"] = std::vector<std::uint8_t>{1, 2};
    meta["i8s"] = std::vector<std::int8_t>{-1, 2};
    meta["u16s"] = std::vector<std::uint16_t>{1, 2};
    meta["i16s"] = std::vector<std::int16_t>{-1, 2};
    meta["i32s"] = std::vector<std::int32_t>{-1, 2};
    meta["u64s"] = std::vector<std::uint64_t>{1, 2};
    meta["i64s"] = std::vector<std::int64_t>{-1, 2};
    meta["floats"] = std::vector<float>{1.5f, 2};
    meta["doubles"] = std::vector<double>{1.5, 2};
    std::vector<TensorRequirement> reqs = {{"base", "tensor_a", "q4_1", {{32, 7, 3, 1}}},
                                           {"base", "tensor_b", "f32", {{4, 1, 1, 1}}}};
    auto p = t.path / "test.gguf";
    gguf(p, meta, reqs);
    auto index = index_tensors(file(p));
    CHECK(index);
    CHECK(index.value().metadata == meta);
    auto found = find_tensor(index.value(), reqs[0]);
    CHECK(found);
    const auto tensor = *found.value();
    CHECK(tensor.bytes == 20 * 7 * 3);
    CHECK(tensor.ne == reqs[0].shape);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> raw(gguf_init_from_file(p.string().c_str(), {true, nullptr}),
                                                            gguf_free);
    CHECK(tensor.offset == gguf_get_data_offset(raw.get()));
    error(read_metadata<float>(meta, "u32"), ErrorCategory::incompatible, "type mismatch");
    error(read_metadata<float>(meta, "absent"), ErrorCategory::incompatible, "missing");
    auto req = reqs[0];
    req.name = "absent";
    error(find_tensor(index.value(), req), ErrorCategory::incompatible, "actual missing");
    req = reqs[0];
    req.type = "f16";
    error(find_tensor(index.value(), req), ErrorCategory::incompatible, "actual q4_1");
    req = reqs[0];
    req.shape[1]++;
    error(find_tensor(index.value(), req), ErrorCategory::incompatible, "actual");
    auto a = locate_slice(tensor, 0, 1), b = locate_slice(tensor, 1, 1);
    CHECK(a && b);
    CHECK(a.value().offset == tensor.offset && a.value().bytes == 140);
    CHECK(b.value().offset == a.value().offset + a.value().bytes);
    error(locate_slice(tensor, 3, 1), ErrorCategory::incompatible, "tensor_a");
    error(locate_slice(tensor, 0, 0), ErrorCategory::incompatible, "tensor_a");
    auto noncontiguous = tensor;
    noncontiguous.nb[2]++;
    error(locate_slice(noncontiguous, 0, 1), ErrorCategory::incompatible, "tensor_a");
    auto original = read(p);
    write(p, "invalid");
    error(index_tensors(file(p)), ErrorCategory::asset_invalid, "test.gguf");
    write(p, original);
    fs::resize_file(p, tensor.offset + tensor.bytes - 1);
    auto truncated = index_tensors(file(p));
    error(truncated, ErrorCategory::asset_invalid, "tensor_a");
    error(truncated, ErrorCategory::asset_invalid, "test.gguf");
    write(p, original);
    auto bad = original;
    auto name = bad.find("tensor_a");
    CHECK(name != std::string::npos);
    auto offset_pos = name + 8 + 4 + 3 * 8 + 4; // GGML serializes three nontrivial dimensions.
    bad[offset_pos] = 1;
    write(p, bad);
    error(index_tensors(file(p)), ErrorCategory::asset_invalid, "test.gguf");
    bad = original;
    auto other = bad.find("tensor_b");
    CHECK(other != std::string::npos);
    bad.replace(other, 8, "tensor_a");
    write(p, bad);
    error(index_tensors(file(p)), ErrorCategory::asset_invalid, "test.gguf");
    bad = original;
    const std::int64_t invalid_width = 31;
    std::memcpy(bad.data() + name + 8 + 4, &invalid_width, sizeof(invalid_width));
    write(p, bad);
    error(index_tensors(file(p)), ErrorCategory::asset_invalid, "test.gguf");
    for (const auto *marker : {"hello", "one", "string"}) {
        auto nul = original;
        auto at = nul.find(marker);
        CHECK(at != std::string::npos);
        nul[at + 1] = '\0';
        write(p, nul);
        error(index_tensors(file(p)), ErrorCategory::asset_invalid, "test.gguf");
    }
    // A valid noncanonical tensor record keeps trailing unit dimensions and
    // arbitrary padding. It must not be rejected by the identity guard.
    gguf(p, {}, {{"base", "x", "f32", {{4, 1, 1, 1}}}});
    const auto canonical = read(p);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> single(
        gguf_init_from_file(p.string().c_str(), {true, nullptr}), gguf_free);
    const auto data_start = gguf_get_data_offset(single.get());
    const auto payload = canonical.substr(data_start);
    auto noncanonical = canonical.substr(0, 57);
    const std::uint64_t one_dimension = 1;
    noncanonical.insert(45, reinterpret_cast<const char *>(&one_dimension), 8);
    const std::uint32_t two_dimensions = 2;
    std::memcpy(noncanonical.data() + 33, &two_dimensions, 4);
    noncanonical.resize(96, static_cast<char>(0xa5));
    write(p, noncanonical + payload);
    auto padded = index_tensors(file(p));
    CHECK(padded);
    CHECK((padded.value().tensors.at("x").ne == std::array<std::int64_t, 4>{{4, 1, 1, 1}}));
    CHECK(padded.value().tensors.at("x").offset == 96);
    auto nul_name = canonical.substr(0, 57);
    nul_name.insert(33, std::string("\0bad", 4));
    const std::uint64_t five_chars = 5;
    std::memcpy(nul_name.data() + 24, &five_chars, 8);
    nul_name.resize(data_start, static_cast<char>(0x55));
    write(p, nul_name + payload);
    error(index_tensors(file(p)), ErrorCategory::asset_invalid, "test.gguf");
    gguf(p, {}, {{"base", "sparse", "f32", {{1024, 1024, 1024, 1}}}});
    CHECK(fs::file_size(p) > 4ULL * 1024 * 1024 * 1024);
#ifndef _WIN32
    model_read_probe_start(p.string().c_str(), fs::file_size(p) - 4ULL * 1024 * 1024 * 1024);
#endif
    auto sparse = index_tensors(file(p));
    CHECK(sparse);
#ifndef _WIN32
    // fread interposition via dlsym(RTLD_NEXT) is POSIX-only; the Windows read-
    // boundary probe awaits a platform-specific port (deferred).
    CHECK(model_read_probe_stop());
#endif
    CHECK(sparse.value().tensors.at("sparse").bytes == 4ULL * 1024 * 1024 * 1024);
    std::cout << "all metadata variants, malformed headers, tensor matching, contiguous slices and 4GiB "
                 "sparse index passed\n";
}
