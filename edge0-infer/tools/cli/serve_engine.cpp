#include "serve_engine.h"
namespace portal {
Engine::Engine(EngineHooks hooks,bool exists):hooks_(std::move(hooks)),session_(exists) {}
Engine::Lease::~Lease() {
    std::lock_guard<std::mutex> lock(owner_.mutex_);
    owner_.active_=false;owner_.last_=std::chrono::steady_clock::now();owner_.idle_.notify_all();
}
std::shared_ptr<Engine::Lease> Engine::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(stopping_)throw Failure(503,"Server shutting down","shutting_down");
    if(active_)throw Failure(429,"Server busy","busy");
    active_=true;
    try {
        const bool rebuilding=session_ && !valid_;
        if(session_ && !valid_) {hooks_.release();session_=false;}
        if(!session_) {hooks_.create();session_=true;valid_=true;}
        if(rebuilding && hooks_.rebuilt)hooks_.rebuilt();
        return std::make_shared<Lease>(*this);
    } catch(...) {active_=false;idle_.notify_all();throw;}
}
edge0::Result<edge0::GenerationResult> Engine::generate(const Request& request,edge0::OutputReceiver receiver) {
    auto result=hooks_.generate(request,std::move(receiver));
    std::lock_guard<std::mutex> lock(mutex_);
    if(result)valid_=result.value().state_valid;
    else if(result.error().category==edge0::ErrorCategory::session_invalid)valid_=false;
    return result;
}
void Engine::expire(double seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(seconds>0 && !active_ && !stopping_ && session_ &&
       std::chrono::duration<double>(std::chrono::steady_clock::now()-last_).count()>=seconds) {
        hooks_.release();session_=false;
    }
}
void Engine::stop() {stopping_=true;hooks_.cancel();}
bool Engine::close() {
    stop();std::unique_lock<std::mutex> lock(mutex_);idle_.wait(lock,[&]{return !active_;});
    if(closed_)return true;
    try {
        if(session_) {hooks_.release();session_=false;}
        hooks_.unload();hooks_.finish();closed_=true;return true;
    } catch(...) {return false;}
}
}
