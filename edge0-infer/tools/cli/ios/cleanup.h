#pragma once
#include <filesystem>
#include <stdexcept>
#include <string>

namespace ios_host {
inline bool valid_invocation(const std::string& name) {
    return !name.empty() && name != "." && name.find("..") == std::string::npos &&
        name.find('/') == std::string::npos && name.find('\\') == std::string::npos;
}

// No concurrent invocation may mutate this sandbox while cleanup is running.
// remove_all removes symlinks themselves; it never traverses their targets.
inline void cleanup(const std::filesystem::path& documents, const std::string& keep, bool clean_only) {
    namespace fs = std::filesystem;
    if ((!clean_only && !valid_invocation(keep)) || (clean_only && !keep.empty()))
        throw std::invalid_argument("Invalid MOE_INVOCATION_DIR");
    const auto base = fs::canonical(documents);
    const auto inv = base / "inv";
    if (fs::is_symlink(fs::symlink_status(inv)))
        throw std::invalid_argument("Invocation root must not be a symlink");
    fs::create_directories(inv);
    if (fs::canonical(inv).parent_path() != base)
        throw std::invalid_argument("Invocation root escapes Documents");
    if (!clean_only) {
        const auto current = fs::symlink_status(inv / keep);
        if (fs::is_symlink(current) || (fs::exists(current) && !fs::is_directory(current)))
            throw std::invalid_argument("Current invocation must be a real directory");
    }
    for (const auto& entry : fs::directory_iterator(inv)) {
        const auto path = fs::absolute(entry.path()).lexically_normal();
        if (path.parent_path() != inv)
            throw std::invalid_argument("Cleanup path escapes invocation root");
        if (!clean_only && path.filename() == keep) {
            continue;
        }
        fs::remove_all(path);
    }
}
}
