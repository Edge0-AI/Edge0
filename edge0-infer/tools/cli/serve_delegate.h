#pragma once
// Delegated models: models that edge0-infer does not compute itself. The Portal
// starts the external engine (edge0-strata) on a loopback port and either proxies
// it (serve) or talks to it as an HTTP client (chat). Their directories carry a
// model.json whose model_name is listed in the registry below, the only place
// that knows their file layout; the native library does not.
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace delegate {
using Json = nlohmann::ordered_json;

// The launch contract version this edge0-infer understands (edge0-strata/edge0/CONTRACT).
constexpr int supported_contract = 4;

struct RegisteredFile {
    std::string path;       // relative to the model directory
    bool directory = false;
    std::string condition;  // empty: always required; otherwise "<option>=<value>"
};
struct Registration {
    std::string id;                          // registration identifier and default alias
    std::string engine;                      // the external engine's directory name
    std::string config_template;             // relative to the engine directory
    std::vector<RegisteredFile> files;
    std::vector<std::pair<std::string, std::vector<std::string>>> options;  // name, allowed values (first = default)
    bool thinking = false, thinking_default = false;
    std::map<std::string, std::string> names;  // model_name -> expert format
};
const std::vector<Registration>& registry();

// A configuration error the CLI reports before starting anything (exit code 2).
struct ConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct FileState {
    const RegisteredFile* file = nullptr;
    bool exists = false;
    std::uintmax_t bytes = 0;
};
struct Match {
    const Registration* registration = nullptr;
    std::vector<FileState> files;
    std::string experts;  // selected by model_name, never a user option
    // Required files (for the given options) that are absent.
    std::vector<std::string> missing(const std::map<std::string, std::string>& options) const;
    // Inspect requires the selected configuration, using the same defaults as serve for omitted options.
    bool complete(const std::map<std::string, std::string>& selected = {}) const;
    // Option choices ("<option>=<value>") that together make an incomplete configuration complete:
    // for each option whose selected value has missing files, the first other value whose files
    // are all present. Empty when any missing file is required by every configuration or an
    // involved option has no complete value, since no option choice can supply it.
    std::vector<std::string> alternatives(const std::map<std::string, std::string>& options) const;
};
// A match when model.json has no architecture and names a registered model; otherwise the native library decides.
std::optional<Match> detect(const std::filesystem::path& directory);
// The directory named by --model, if any.
std::optional<std::string> model_argument(int argc, char** argv);
Json inspect_json(const Match&, const std::filesystem::path& directory,
                  const std::map<std::string, std::string>& selected = {});

// Options shared by serve and chat in delegate mode.
enum class Platform { posix, windows };
constexpr Platform host_platform() {
#ifdef _WIN32
    return Platform::windows;
#else
    return Platform::posix;
#endif
}

struct LaunchOptions {
    std::filesystem::path model, strata_dir, python;
    std::string alias;
    std::string experts;  // copied from Match for template expansion
    Platform platform = host_platform();
    std::vector<std::filesystem::path> lib_dirs;
    std::vector<std::string> warnings;
    int gpu = 0;
    std::optional<std::int64_t> context;
    std::map<std::string, std::string> model_options;  // validated and completed with defaults
    std::vector<std::string> strata_args;  // engine command-line words appended after the template's args
    double ready_timeout = 1800;
};
int parse_device(const std::string& value);  // "cuda" or "cuda:<n>" -> n; otherwise ConfigError
// Validates names and values against the registration and fills defaults.
std::map<std::string, std::string> resolve_options(const Registration&, const std::map<std::string, std::string>& given);
// The engine directory: --strata-dir, then EDGE0_STRATA_DIR, then an edge0-strata/ beside an ancestor of the executable.
std::filesystem::path locate_engine(const std::optional<std::string>& given, const std::filesystem::path& executable);
std::filesystem::path executable_path(const std::string& argv0);
// Reads <engine>/edge0/CONTRACT; throws ConfigError naming both versions on a mismatch.
void check_contract(const std::filesystem::path& engine);
// Expands the registration's template into the external engine's config. options.strata_args are appended to
// its args last; one equal to --max-context, or an option ("--...") the expanded template already sets, is a ConfigError.
Json expand_template(const Json& config_template, const Registration&, const LaunchOptions&, Platform);
Json load_template(const std::filesystem::path& engine, const Registration&);

// Command-line arguments shared by serve and chat in delegate mode.
struct CommonArgs {
    std::string model, device = "cuda:0", alias;
    std::optional<std::string> prefill_device, strata_dir, python;
    std::map<std::string, std::string> model_options;
    std::optional<std::int64_t> context;
    std::vector<std::string> strata_args;  // --strata-arg, in order
    double ready_timeout = 1800;
};
// True when the key is one of the shared value options (and stores it).
bool common_argument(const std::string& key, const std::string& value, CommonArgs&);
// An option of the native engine that delegate mode refuses; throws ConfigError.
[[noreturn]] void unsupported_option(const std::string& key);
// Validates device, options and files; locates the engine and its Python.
LaunchOptions resolve(const CommonArgs&, const Match&, const std::filesystem::path& executable, Platform,
                      const std::string& cuda_path);

// The external engine's server process.
class Engine {
public:
    struct Settings {
        std::filesystem::path python, server_script, config, ready_file, log_file;  // empty log_file: inherit stderr
        std::vector<std::filesystem::path> lib_dirs;
        double ready_timeout = 1800;
        const std::atomic<bool>* interrupted = nullptr;  // checked while waiting for readiness
    };
    explicit Engine(Settings);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    // Starts the process and waits for the ready file and a healthy /health. Throws on failure, after stopping it.
    void start();
    int port() const { return port_; }
    long pid() const { return pid_; }
    bool running() const { return !exited_; }
    std::optional<int> exit_code() const;
    // SIGTERM to the process group, then SIGKILL after the grace period; waits until it has exited.
    // Safe to call from several threads: the first call stops the engine, the others wait for it and return.
    void stop(double grace_seconds);
private:
    Settings settings_;
#ifdef _WIN32
    struct WindowsState;
    std::unique_ptr<WindowsState> windows_;
#endif
    long pid_ = 0;
    int port_ = 0;
    std::atomic<bool> exited_{false};
    std::atomic<int> code_{0};
    std::thread monitor_;
    std::mutex stop_mutex_;  // makes stop() run once at a time (monitor_ is joined only once)
    void signal_group(bool force);
};

// A private temporary directory removed on destruction.
struct TempDir {
    std::filesystem::path path;
    TempDir();
    ~TempDir();
    // Removes the directory now (also safe before std::_Exit); later calls do nothing.
    void remove();
    // Leaves the directory in place (for a log the user is pointed to).
    void keep() { path.clear(); }
};

// One complete launch: options -> engine directory -> config -> running engine.
struct Launch {
    TempDir temp;
    std::filesystem::path engine_dir;
    Json config;
    std::unique_ptr<Engine> engine;
};
std::unique_ptr<Launch> launch(const Registration&, const LaunchOptions&, const std::filesystem::path& log_file,
                               const std::atomic<bool>* interrupted = nullptr);

// Subcommands other than serve, chat and inspect refuse delegated directories.
int refuse(const std::string& command, const Match&);
int serve(int argc, char** argv, const Match&);
int chat(int argc, char** argv, const Match&);
int inspect(int argc, char** argv, const Match&);
// Dispatches a delegated directory; returns nothing when the directory is not delegated.
std::optional<int> dispatch(const std::string& command, int argc, char** argv);
}
