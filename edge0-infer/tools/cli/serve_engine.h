#pragma once
#include "serve_protocol.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
namespace portal {
// One synchronous handoff: the producer cannot run past its current event until
// the HTTP provider takes over. Early native errors remain ordinary HTTP errors.
template<class Sink> struct StreamHandoff {
    std::mutex mutex;
    std::condition_variable changed;
    Sink* sink=nullptr;
    bool ready=false,attached=false,aborted=false,done=false;
    std::exception_ptr error;
    std::thread worker;
    Sink* await_sink() {
        std::unique_lock<std::mutex> lock(mutex);ready=true;changed.notify_all();
        changed.wait(lock,[&]{return attached || aborted;});return aborted?nullptr:sink;
    }
    void finish(std::exception_ptr failure) {
        std::lock_guard<std::mutex> lock(mutex);error=failure;done=true;changed.notify_all();
    }
    void wait_ready() {
        std::unique_lock<std::mutex> lock(mutex);changed.wait(lock,[&]{return ready || done;});
        if(done && error)std::rethrow_exception(error);
    }
    void attach(Sink& value) {
        std::lock_guard<std::mutex> lock(mutex);sink=&value;attached=true;changed.notify_all();
    }
    void abort() {
        {std::lock_guard<std::mutex> lock(mutex);aborted=true;changed.notify_all();}
        if(worker.joinable())worker.join();
    }
    ~StreamHandoff(){abort();}
};

struct EngineHooks {
    std::function<void()> create, release, unload, finish, cancel, rebuilt;
    std::function<edge0::Result<edge0::GenerationResult>(const Request&,edge0::OutputReceiver)> generate;
};
class Engine {
public:
    explicit Engine(EngineHooks hooks, bool session_exists = true);
    class Lease {
    public:
        explicit Lease(Engine& owner): owner_(owner) {}
        ~Lease();
        Lease(const Lease&)=delete;
        Lease& operator=(const Lease&)=delete;
    private: Engine& owner_;
    };
    std::shared_ptr<Lease> acquire();
    edge0::Result<edge0::GenerationResult> generate(const Request&,edge0::OutputReceiver);
    void expire(double seconds);
    void stop();
    bool close();
    bool stopping() const { return stopping_.load(); }
private:
    EngineHooks hooks_;
    std::mutex mutex_;
    std::condition_variable idle_;
    bool active_=false,session_=true,valid_=true,closed_=false;
    std::atomic<bool> stopping_{false};
    std::chrono::steady_clock::time_point last_=std::chrono::steady_clock::now();
};
}
