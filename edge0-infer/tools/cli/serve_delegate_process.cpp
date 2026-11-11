#include "serve_delegate.h"
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <httplib.h>
#include <csignal>
#include <cerrno>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

// The external engine is started through the standard popen() with a small Python
// launcher (the engine's own interpreter). The launcher reports its process ID on
// the pipe, puts itself in a new process group (POSIX) and moves its output away
// from the pipe. On POSIX it then forks: the child becomes the engine's server and
// the launcher stays as its supervisor. The supervisor exits with the server's
// status, so pclose() on a monitor thread returns exactly when the server exits;
// and when edge0-infer ends without stopping it (kill -9, a crash), the supervisor
// sees its parent change and stops the server, then the whole process group.
// (Linux's parent-death signal was not reliable for this: the kernel ties it to a
// thread, and in tests a server sometimes outlived a killed edge0-infer.)
namespace delegate {
namespace fs = std::filesystem;
namespace {
#ifndef _WIN32
const char* launcher_source =
    "import os, signal, sys, time\n"
    "os.environ.pop('STRATA_API_KEY', None)\n"
    "log = sys.argv[1]\n"
    "if os.name != 'nt':\n"
    "    os.setpgid(0, 0)\n"
    "sys.stdout.write('edge0-launcher-pid ' + str(os.getpid()) + '\\n')\n"
    "sys.stdout.flush()\n"
    "null = os.open(os.devnull, os.O_RDONLY)\n"
    "os.dup2(null, 0)\n"
    "os.close(null)\n"
    "if log != '-':\n"
    "    out = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)\n"
    "    os.dup2(out, 1)\n"
    "    os.dup2(out, 2)\n"
    "    os.close(out)\n"
    "else:\n"
    "    os.dup2(2, 1)\n"
    "if os.name == 'nt':\n"
    "    import runpy\n"
    "    sys.argv = sys.argv[2:]\n"
    "    sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))\n"
    "    runpy.run_path(sys.argv[0], run_name='__main__')\n"
    "    sys.exit(0)\n"
    "parent = os.getppid()\n"
    "child = os.fork()\n"
    "if child == 0:\n"
    "    try:\n"
    "        os.execv(sys.executable, [sys.executable] + sys.argv[2:])\n"
    "    finally:\n"
    "        os._exit(127)\n"
    "signal.signal(signal.SIGTERM, lambda *_: None)  # edge0-infer stops the group; wait for the server's own exit\n"
    "orphaned = None\n"
    "while True:\n"
    "    pid, status = os.waitpid(child, os.WNOHANG)\n"
    "    if pid == child:\n"
    "        break\n"
    "    if orphaned is None and os.getppid() != parent:\n"
    "        orphaned = time.monotonic()\n"
    "        os.kill(child, signal.SIGTERM)\n"
    "    elif orphaned is not None and time.monotonic() - orphaned > 60:\n"
    "        os.killpg(0, signal.SIGKILL)\n"
    "    time.sleep(0.1)\n"
    "if orphaned is not None:\n"
    "    os.killpg(0, signal.SIGKILL)  # nobody is left to stop what the server leaves behind\n"
    "if os.WIFSIGNALED(status):\n"
    "    if os.WTERMSIG(status) not in (signal.SIGKILL, signal.SIGSTOP):  # those two cannot be set, and act by default\n"
    "        signal.signal(os.WTERMSIG(status), signal.SIG_DFL)\n"
    "    os.kill(os.getpid(), os.WTERMSIG(status))\n"
    "os._exit(os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1)\n";

std::string quote(const std::string& value) {
    std::string out = "'";
    for (char c : value) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return out + "'";
}

int decode_status(int status) {
    if ((status & 0x7f) == 0) return (status >> 8) & 0xff;  // exited normally
    return 128 + (status & 0x7f);                            // ended by a signal (shell convention)
}

#endif

void write_file(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
    output.close();
    if (!output) throw std::runtime_error("Cannot write " + path.string());
}
}

fs::path executable_path(const std::string& argv0) {
#ifdef _WIN32
    (void)argv0;
    std::vector<wchar_t> buffer(32768);
    const auto length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (!length || length >= buffer.size()) throw ConfigError("Cannot locate edge0-infer executable");
    return fs::path(std::wstring(buffer.data(), length));
#else
    std::error_code error;
    fs::path executable = fs::read_symlink("/proc/self/exe", error);
    if (error || executable.empty()) executable = fs::absolute(argv0, error);
    return executable;
#endif
}

TempDir::TempDir() {
    std::random_device random;
    for (int attempt = 0; attempt < 16; ++attempt) {
        std::ostringstream name;
        name << "edge0-strata-" << std::hex << random() << random();
        const auto candidate = fs::temp_directory_path() / name.str();
        std::error_code error;
        if (fs::create_directory(candidate, error)) {
            fs::permissions(candidate, fs::perms::owner_all, fs::perm_options::replace, error);
            path = candidate;
            return;
        }
    }
    throw std::runtime_error("Cannot create a private temporary directory");
}

TempDir::~TempDir() { remove(); }

void TempDir::remove() {
    std::error_code error;
    if (!path.empty()) fs::remove_all(path, error);
    path.clear();
}

#ifdef _WIN32
namespace {
struct Handle {
    HANDLE value = nullptr;
    Handle() = default;
    explicit Handle(HANDLE handle) : value(handle) {}
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};
[[noreturn]] void win_error(const char* operation) {
    throw std::runtime_error(std::string(operation) + " failed (Windows error " + std::to_string(GetLastError()) + ")");
}
std::wstring quote_windows(const std::wstring& value) {
    std::wstring out = L"\"";
    std::size_t slashes = 0;
    for (const auto c : value) {
        if (c == L'\\') { ++slashes; continue; }
        out.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
        out += c;
        slashes = 0;
    }
    out.append(slashes * 2, L'\\');
    return out + L"\"";
}
std::vector<wchar_t> child_environment(const std::vector<fs::path>& directories) {
    struct IgnoreCase {
        bool operator()(const std::wstring& a, const std::wstring& b) const { return _wcsicmp(a.c_str(), b.c_str()) < 0; }
    };
    std::map<std::wstring, std::wstring, IgnoreCase> values;
    auto block = GetEnvironmentStringsW();
    if (!block) win_error("GetEnvironmentStringsW");
    try {
        for (auto entry = block; *entry; entry += std::wcslen(entry) + 1) {
            const std::wstring line(entry);
            const auto split = line.find(L'=', 1);  // preserve the hidden =C: drive variables too
            if (split != std::wstring::npos) values.emplace(line.substr(0, split), line.substr(split + 1));
        }
    } catch (...) { FreeEnvironmentStringsW(block); throw; }
    FreeEnvironmentStringsW(block);
    values.erase(L"STRATA_API_KEY");
    if (!directories.empty()) {
        std::wstring prefix;
        for (const auto& directory : directories) prefix += directory.wstring() + L";";
        values[L"PATH"] = prefix + values[L"PATH"];
    }
    std::vector<wchar_t> result;
    for (const auto& entry : values) {
        const auto line = entry.first + L"=" + entry.second;
        result.insert(result.end(), line.begin(), line.end());
        result.push_back(L'\0');
    }
    result.push_back(L'\0');
    if (values.empty()) result.push_back(L'\0');
    return result;
}
}
struct Engine::WindowsState {
    Handle job, process;
    ~WindowsState() {
        if (job.value) TerminateJobObject(job.value, 1);
        if (process.value) WaitForSingleObject(process.value, INFINITE);
    }
};
#endif

Engine::Engine(Settings settings) : settings_(std::move(settings)) {}

Engine::~Engine() {
    try { stop(10); } catch (...) {}
}

std::optional<int> Engine::exit_code() const {
    if (!exited_) return std::nullopt;
    return code_.load();
}

void Engine::signal_group(bool force) {
    if (pid_ <= 0 || exited_) return;
#ifdef _WIN32
    if (force) TerminateJobObject(windows_->job.value, 1);
    else if (!GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, static_cast<DWORD>(pid_))) {
        std::cerr << "edge0-strata: Ctrl-Break unavailable (Windows error " << GetLastError()
                  << "); terminating the process job\n";
        TerminateJobObject(windows_->job.value, 1);
    }
#else
    const int sig = force ? SIGKILL : SIGTERM;
    if (::kill(static_cast<pid_t>(-pid_), sig) != 0) ::kill(static_cast<pid_t>(pid_), sig);
#endif
}

void Engine::start() {
#ifdef _WIN32
    windows_ = std::make_unique<WindowsState>();
    windows_->job.value = CreateJobObjectW(nullptr, nullptr);
    if (!windows_->job.value) win_error("CreateJobObjectW");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(windows_->job.value, JobObjectExtendedLimitInformation, &limits, sizeof limits))
        win_error("SetInformationJobObject");
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    Handle input(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, 0, nullptr));
    if (input.value == INVALID_HANDLE_VALUE) win_error("Open NUL");
    Handle output;
    if (settings_.log_file.empty()) {
        if (!DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE), GetCurrentProcess(),
                             &output.value, 0, TRUE, DUPLICATE_SAME_ACCESS)) win_error("Duplicate stderr");
    } else {
        output.value = CreateFileW(settings_.log_file.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                   &security, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (output.value == INVALID_HANDLE_VALUE) win_error("Open edge0-strata log");
    }
    // Inherit only stdio, never a job or another thread's pipe endpoints.
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<unsigned char> storage(bytes);
    auto attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    if (!InitializeProcThreadAttributeList(attributes, 1, 0, &bytes)) win_error("InitializeProcThreadAttributeList");
    struct Attributes { LPPROC_THREAD_ATTRIBUTE_LIST value; ~Attributes() { DeleteProcThreadAttributeList(value); } } cleanup{attributes};
    HANDLE inherited[] = {input.value, output.value};
    if (!UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited, sizeof inherited, nullptr, nullptr))
        win_error("UpdateProcThreadAttribute");
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof startup;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = input.value;
    startup.StartupInfo.hStdOutput = startup.StartupInfo.hStdError = output.value;
    startup.lpAttributeList = attributes;
    const auto command_text = quote_windows(settings_.python.wstring()) + L" " + quote_windows(settings_.server_script.wstring()) +
        L" --engine strata --config " + quote_windows(settings_.config.wstring()) +
        L" --host 127.0.0.1 --port 0 --ready-file " + quote_windows(settings_.ready_file.wstring());
    auto command = command_text;
    auto environment = child_environment(settings_.lib_dirs);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(settings_.python.c_str(), command.data(), nullptr, nullptr, TRUE,
                        CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                        environment.data(), nullptr, &startup.StartupInfo, &process)) win_error("CreateProcessW");
    windows_->process.value = process.hProcess;
    Handle thread(process.hThread);
    if (!AssignProcessToJobObject(windows_->job.value, process.hProcess)) {
        const auto error = GetLastError();
        TerminateProcess(process.hProcess, 1);
        WaitForSingleObject(process.hProcess, INFINITE);
        SetLastError(error);
        win_error("AssignProcessToJobObject");
    }
    pid_ = static_cast<long>(process.dwProcessId);
    if (ResumeThread(thread.value) == static_cast<DWORD>(-1)) win_error("ResumeThread");
    monitor_ = std::thread([this] {
        const auto waited = WaitForSingleObject(windows_->process.value, INFINITE);
        DWORD code = 1;
        if (waited == WAIT_OBJECT_0) GetExitCodeProcess(windows_->process.value, &code);
        code_ = static_cast<int>(code);
        exited_ = true;
    });
#else
    const auto launcher = settings_.ready_file.parent_path() / "launch.py";
    write_file(launcher, launcher_source);
    std::string command = quote(settings_.python.string()) + " " + quote(launcher.string()) + " " +
                          quote(settings_.log_file.empty() ? std::string("-") : settings_.log_file.string()) + " " +
                          quote(settings_.server_script.string()) + " --engine strata --config " + quote(settings_.config.string()) +
                          " --host 127.0.0.1 --port 0 --ready-file " + quote(settings_.ready_file.string());
    FILE* pipe = popen(("exec " + command).c_str(), "r");
    if (!pipe) throw std::runtime_error("Cannot start edge0-strata");
    std::mutex mutex;
    std::condition_variable announced;
    bool have_pid = false;
    std::string first_line;
    monitor_ = std::thread([this, pipe, &mutex, &announced, &have_pid, &first_line] {
        std::string line;
        bool first = true, oversized = false;
        long reported = 0;
        const std::string prefix = "edge0-launcher-pid ";
        for (;;) {
            const int c = std::fgetc(pipe);
            if (c != EOF && c != '\n') {
                if (line.size() < 128) line += static_cast<char>(c);
                else oversized = true;
                continue;
            }
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (first) {first_line = line.substr(0, 64);first = false;}
            if (!oversized && line.compare(0, prefix.size(), prefix) == 0) {
                const auto digits = line.substr(prefix.size());
                if (!digits.empty() && digits.find_first_not_of("0123456789") == std::string::npos) {
                    errno = 0;
                    const long value = std::strtol(digits.c_str(), nullptr, 10);
                    if (errno != ERANGE && value > 0 && value <= std::numeric_limits<pid_t>::max()) reported = value;
                }
            }
            if (reported > 0 || c == EOF) break;
            line.clear();oversized = false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            pid_ = reported;
            have_pid = true;
            announced.notify_all();
        }
        const int status = pclose(pipe);
        code_ = decode_status(status);
        exited_ = true;
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        announced.wait(lock, [&] { return have_pid; });
    }
    if (pid_ <= 0) {
        monitor_.join();
        throw std::runtime_error("Cannot start edge0-strata: no launcher PID from --strata-python; first output: " + first_line);
    }
#endif
    const auto started = std::chrono::steady_clock::now();
    auto fail = [&](const std::string& message) {
        stop(10);
        throw std::runtime_error(message);
    };
    for (;;) {
        if (settings_.interrupted && settings_.interrupted->load()) fail("interrupted while edge0-strata was starting");
        if (exited_)
            fail("edge0-strata exited before it was ready (exit code " + std::to_string(code_.load()) + ")" +
                 (settings_.log_file.empty() ? "" : "; see " + settings_.log_file.string()));
        std::error_code error;
        if (port_ == 0 && fs::exists(settings_.ready_file, error)) {
            std::ifstream input(settings_.ready_file, std::ios::binary);
            const auto record = Json::parse(input, nullptr, false);
            if (!record.is_discarded() && record.contains("bound_port") && record["bound_port"].is_number_integer())
                port_ = record["bound_port"].get<int>();
        }
        if (port_ > 0) {
            httplib::Client client("127.0.0.1", port_);
            client.set_tcp_nodelay(true);
            client.set_connection_timeout(2);
            client.set_read_timeout(5);
            if (auto response = client.Get("/health"); response && response->status == 200) return;
        }
        const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (waited > settings_.ready_timeout)
            fail("edge0-strata was not ready within " + std::to_string(static_cast<long long>(settings_.ready_timeout)) + " s" +
                 (settings_.log_file.empty() ? "" : "; see " + settings_.log_file.string()));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void Engine::stop(double grace_seconds) {
    std::lock_guard<std::mutex> once(stop_mutex_);
    if (!monitor_.joinable()) return;
    if (!exited_) {
        signal_group(false);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(grace_seconds);
        while (!exited_ && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (!exited_) signal_group(true);
    }
    monitor_.join();
#ifdef _WIN32
    windows_.reset();
#endif
#ifndef _WIN32
    // The launcher's process group may still hold children that outlived it. Known limit: the launcher has
    // been reaped at this point, so if the group is already empty and the system has reused its number for a
    // new process group in the meantime, that group would receive the SIGKILL. The window is a few
    // microseconds; closing it needs the launcher itself to clear its group before it exits (or a
    // platform-specific ownership check), which this tool's portable process code does not have.
    if (pid_ > 0) ::kill(static_cast<pid_t>(-pid_), SIGKILL);
#endif
}

std::unique_ptr<Launch> launch(const Registration& registration, const LaunchOptions& options, const fs::path& log_file,
                               const std::atomic<bool>* interrupted) {
    auto result = std::make_unique<Launch>();
    result->engine_dir = options.strata_dir;
    check_contract(options.strata_dir);
    const auto server = options.strata_dir / "serve" / "server.py";
    std::error_code error;
    if (!fs::is_regular_file(server, error)) throw ConfigError("edge0-strata is incomplete: " + server.string() + " is missing");
    if (!fs::is_regular_file(options.python, error))
        throw ConfigError("Python for edge0-strata not found: " + options.python.string() +
                          " (create edge0-strata/.venv as edge0-strata/EDGE0.md describes, or pass --strata-python)");
    result->config = expand_template(load_template(options.strata_dir, registration), registration, options, options.platform);
    const auto config = result->temp.path / "config.json";
    write_file(config, result->config.dump(2) + "\n");
    Engine::Settings settings;
    settings.python = options.python;
    settings.lib_dirs = options.lib_dirs;
    settings.server_script = server;
    settings.config = config;
    settings.ready_file = result->temp.path / "ready.json";
    settings.log_file = log_file;
    settings.ready_timeout = options.ready_timeout;
    settings.interrupted = interrupted;
    result->engine = std::make_unique<Engine>(settings);
    result->engine->start();
    return result;
}
}
