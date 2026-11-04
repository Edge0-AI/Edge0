#include "app/text/rules.h"
#include "app/adapters.h"
#include <limits>
#include "model/sha256.h"
namespace edge0::text {
namespace {
struct MetadataError { edge0::Error error; };
template<class T> const T &get(const model::Metadata &metadata, const std::string &key) {
    auto result = model::read_metadata<T>(metadata, key);
    if (!result) throw MetadataError{result.error()};
    return *result.value();
}
}
Result<Rules> load_rules(const model::AssetView &assets, const model::Adapter &adapter) {
    try {
        auto base = assets.find("base");
        if (base == assets.end())
            return edge0::Error{ErrorCategory::incompatible, "missing GGUF role base"};
        const auto &m = base->second.metadata;
        TokenData d;
        d.model = get<std::string>(m, "tokenizer.ggml.model");
        d.pre = get<std::string>(m, "tokenizer.ggml.pre");
        d.tokens = get<std::vector<std::string>>(m, "tokenizer.ggml.tokens");
        d.types = get<std::vector<std::int32_t>>(m, "tokenizer.ggml.token_type");
        d.merges = get<std::vector<std::string>>(m, "tokenizer.ggml.merges");
        if (d.tokens.size() != d.types.size())
            return edge0::Error{ErrorCategory::incompatible, "tokenizer.ggml.token_type: token count mismatch"};
        auto hex_digit = [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        };
        for (std::size_t i = 0; i < d.tokens.size(); ++i) {
            const auto &token = d.tokens[i];
            const auto key = "tokenizer.ggml.tokens[" + std::to_string(i) + "]: ";
            if (!valid_utf8(token))
                return edge0::Error{ErrorCategory::incompatible, key + "invalid UTF-8"};
            // Byte spellings are interpreted later by token_to_piece. Validate the
            // complete spelling here, before malformed asset data reaches that path.
            if (d.types[i] == 6 &&
                (token.size() != 6 || token.compare(0, 3, "<0x") != 0 || token[5] != '>' ||
                 !hex_digit(token[3]) || !hex_digit(token[4])))
                return edge0::Error{ErrorCategory::incompatible, key + "invalid byte token; expected <0xHH>"};
        }
        auto id = [&](const char *key) -> std::int32_t {
            if (!m.count(key)) return -1;
            auto value = get<std::uint32_t>(m, key);
            if (value > std::uint32_t(std::numeric_limits<std::int32_t>::max()))
                throw MetadataError{{ErrorCategory::incompatible, std::string(key) + ": token ID overflow"}};
            return static_cast<std::int32_t>(value);
        };
        auto flag = [&](const char *key) { return m.count(key) ? get<bool>(m, key) : false; };
        d.bos = id("tokenizer.ggml.bos_token_id");
        d.eos = id("tokenizer.ggml.eos_token_id");
        d.eot = id("tokenizer.ggml.eot_token_id");
        d.eom = id("tokenizer.ggml.eom_token_id");
        d.padding = id("tokenizer.ggml.padding_token_id");
        d.add_bos = flag("tokenizer.ggml.add_bos_token");
        d.add_eos = flag("tokenizer.ggml.add_eos_token");
        model::Sha256 hash;
        const auto add = [&](const std::string& value) { hash.update(std::to_string(value.size()) + ":" + value); };
        add(d.model); add(d.pre);
        add("tokens"); add(std::to_string(d.tokens.size()));
        for (const auto& token : d.tokens) add(token);
        add("types"); add(std::to_string(d.types.size()));
        for (const auto type : d.types) add(std::to_string(type));
        add("merges"); add(std::to_string(d.merges.size()));
        for (const auto& merge : d.merges) add(merge);
        for (const auto value : {d.bos, d.eos, d.eot, d.eom, d.padding}) add(std::to_string(value));
        add(d.add_bos ? "1" : "0"); add(d.add_eos ? "1" : "0");
        return Rules{Bpe(std::move(d)), adapter.describe_text(), &adapter, hash.finish()};
    } catch (const MetadataError &e) { return e.error; }
      catch (const TextError &e) { return edge0::Error{ErrorCategory::incompatible, e.what()}; }
}
Result<Rules> load_rules(const std::string &directory) {
    auto catalog = model::open_catalog(directory);
    if (!catalog) return catalog.error();
    auto adapter = model::select_adapter(catalog.value(), app::compiled_adapters());
    if (!adapter) return adapter.error();
    auto compatible = model::check_compatibility(catalog.value(), *adapter.value());
    if (!compatible) return compatible.error();
    return load_rules(compatible.value().indexes, *adapter.value());
}
} // namespace edge0::text
