#include "tools/cli/ios/cleanup.h"
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("moe-ios-cleanup-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(root / "inv/current");
        fs::create_directories(root / "inv/old/nested");
        fs::create_directories(root / "assets");
        std::ofstream(root / "assets/keep") << "keep";
        std::ofstream(root / "inv/current/input") << "input";
        std::ofstream(root / "inv/old/nested/file") << "old";
        fs::create_directory_symlink(root / "assets", root / "inv/link");
        fs::create_directory_symlink(root / "assets", root / "inv/old/nested/link");
        for (const auto& invalid : {"", "..", ".", "../assets", "a/b", "a\\b", "a..b"}) {
            bool rejected = false;
            try { ios_host::cleanup(root, invalid, false); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Invalid invocation accepted");
        }
        bool wrong_mode = false;
        try { ios_host::cleanup(root, "current", true); }
        catch (const std::invalid_argument&) { wrong_mode = true; }
        if (!wrong_mode) throw std::runtime_error("Clean-only accepted a preserved directory");
        fs::create_directory_symlink(root / "assets", root / "inv/bad-current");
        bool linked_current = false;
        try { ios_host::cleanup(root, "bad-current", false); }
        catch (const std::invalid_argument&) { linked_current = true; }
        if (!linked_current || !fs::exists(root / "assets/keep"))
            throw std::runtime_error("Symlink current invocation accepted");
        if (!fs::exists(root / "inv/current/input") || !fs::exists(root / "inv/old/nested/file"))
            throw std::runtime_error("Rejected invocation mutated old entries");
        ios_host::cleanup(root, "current", false);
        if (!fs::exists(root / "inv/current/input") || fs::exists(root / "inv/old") ||
            fs::is_symlink(fs::symlink_status(root / "inv/link")) || !fs::exists(root / "assets/keep"))
            throw std::runtime_error("Cleanup violated boundaries");
        ios_host::cleanup(root, "", true);
        if (!fs::is_empty(root / "inv") || !fs::exists(root / "assets/keep"))
            throw std::runtime_error("Clean-only violated boundaries");
        fs::remove(root / "inv");
        fs::create_directory_symlink(root / "assets", root / "inv");
        bool rejected = false;
        try { ios_host::cleanup(root, "", true); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || !fs::exists(root / "assets/keep")) throw std::runtime_error("Symlink root accepted");
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        fs::remove_all(root);
        return 1;
    }
}
