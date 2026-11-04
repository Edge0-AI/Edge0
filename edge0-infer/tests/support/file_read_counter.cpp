#ifdef _WIN32
// fread/open/pread interposition via dlsym(RTLD_NEXT) is POSIX-only. The Windows
// engine reads through ReadFile, not stdio; a platform-specific port awaits.
// Provide stubs so dependent tests link; callers skip the read-count assertions
// on Windows (the counter reports 0 and start/stop are no-ops).
extern "C" void file_read_counter_start() {}
extern "C" unsigned file_read_counter_stop() { return 0; }
#else
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
namespace {
std::atomic<bool> tracking{false};
std::atomic<unsigned> calls{0};
void count() { if(tracking) ++calls; }
template<class Function> Function original(const char* name) {
    auto function=reinterpret_cast<Function>(dlsym(RTLD_NEXT,name));
    if(!function) std::abort();
    return function;
}
}
extern "C" void file_read_counter_start() { calls=0; tracking=true; }
extern "C" unsigned file_read_counter_stop() { tracking=false; return calls.load(); }
extern "C" size_t fread(void* out,size_t size,size_t elements,FILE* file) {
    static auto function=original<size_t (*)(void*,size_t,size_t,FILE*)>("fread");
    count();return function(out,size,elements,file);
}
extern "C" FILE* fopen(const char* path,const char* mode) {
    static auto function=original<FILE* (*)(const char*,const char*)>("fopen");
    count();return function(path,mode);
}
extern "C" int open(const char* path,int flags,...) {
    static auto function=original<int (*)(const char*,int,...)>("open");
    count();
    bool mode_required=(flags&O_CREAT)!=0;
#ifdef O_TMPFILE
    mode_required=mode_required || (flags&O_TMPFILE)==O_TMPFILE;
#endif
    if(mode_required) {
        va_list arguments;va_start(arguments,flags);
        const auto mode=va_arg(arguments,int);va_end(arguments);
        return function(path,flags,mode);
    }
    return function(path,flags);
}
extern "C" ssize_t read(int file,void* out,size_t size) {
    static auto function=original<ssize_t (*)(int,void*,size_t)>("read");
    count();return function(file,out,size);
}
extern "C" ssize_t pread(int file,void* out,size_t size,off_t offset) {
    static auto function=original<ssize_t (*)(int,void*,size_t,off_t)>("pread");
    count();return function(file,out,size,offset);
}
#endif
