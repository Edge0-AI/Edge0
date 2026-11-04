#include <UIKit/UIKit.h>
#include <Foundation/Foundation.h>
#include <os/proc.h>
#include <pthread.h>
#include <unistd.h>
#include <csignal>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include "cleanup.h"
#include "chat.h"

int moe_cli_main(int, char**);
static int cli_argc;
static char** cli_argv;

static void probe(NSString* path, NSString* phase) {
    if (!path) return;
    NSArray* states = @[@"nominal", @"fair", @"serious", @"critical"];
    const auto state = [NSProcessInfo processInfo].thermalState;
    NSDictionary* snapshot = @{@"thermal_state": states[static_cast<NSUInteger>(state)],
        @"os_proc_available_memory": @(os_proc_available_memory()),
        @"physical_memory": @([NSProcessInfo processInfo].physicalMemory),
        @"timestamp": @([[NSDate date] timeIntervalSince1970])};
    NSMutableDictionary* data = [NSMutableDictionary dictionary];
    NSData* old = [NSData dataWithContentsOfFile:path];
    if (old) {
        id parsed = [NSJSONSerialization JSONObjectWithData:old options:0 error:nil];
        if ([parsed isKindOfClass:[NSDictionary class]]) [data addEntriesFromDictionary:parsed];
    }
    data[phase] = snapshot;
    NSData* encoded = [NSJSONSerialization dataWithJSONObject:data options:0 error:nil];
    if (![encoded writeToFile:path atomically:YES]) throw std::runtime_error("Cannot write host probe");
}

static NSString* file_env(const char* key, const std::string& invocation) {
    const char* value = std::getenv(key);
    if (!value || !*value) return nil;
    const std::filesystem::path path(value);
    if (path.is_absolute() || path.lexically_normal() != path ||
        path.parent_path() != std::filesystem::path("inv") / invocation)
        throw std::invalid_argument("Host file must belong to the current invocation");
    if (std::filesystem::is_symlink(std::filesystem::symlink_status(path)))
        throw std::invalid_argument("Host file must not be a symlink");
    return [NSString stringWithUTF8String:value];
}

// The CLI owns a full-size stack and unblocked signals inherited by its helper threads.
static void* run_cli(void*) {
    int code = 1;
    @autoreleasepool {
        try {
            sigset_t signals;
            if (sigemptyset(&signals) || sigaddset(&signals, SIGINT) || sigaddset(&signals, SIGTERM) ||
                pthread_sigmask(SIG_UNBLOCK, &signals, nullptr))
                throw std::runtime_error("Cannot unblock CLI termination signals");
            NSString* documents = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
            std::filesystem::current_path(documents.UTF8String);
            const char* value = std::getenv("MOE_INVOCATION_DIR");
            const std::string invocation = value ? value : "";
            const bool clean_only = std::getenv("MOE_HOST_CLEAN_ONLY") != nullptr;
            // Direct CLI launches without wrapper variables have no temporary files to clean.
            if (value || clean_only) ios_host::cleanup(documents.UTF8String, invocation, clean_only);
            if (clean_only) code = 0;
            else {
                NSString* input = file_env("MOE_STDIN_FILE", invocation);
                if (!std::freopen(input ? input.UTF8String : "/dev/null", "rb", stdin))
                    throw std::runtime_error("Cannot open host stdin");
                NSString* errors = file_env("MOE_STDERR_FILE", invocation);
                if (errors && !std::freopen(errors.UTF8String, "wb", stderr))
                    throw std::runtime_error("Cannot open host stderr");
                NSString* snapshot = file_env("MOE_HOST_PROBE", invocation);
                probe(snapshot, @"before");
                code = moe_cli_main(cli_argc, cli_argv);
                probe(snapshot, @"after");
            }
        } catch (const std::exception& error) {
            code = 1;
            std::fprintf(stderr, "iOS host: %s\n", error.what());
        }
    }
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    _exit(code);
}

static void start_cli() {
    pthread_attr_t attributes;
    int error = pthread_attr_init(&attributes);
    if (!error) {
        error = pthread_attr_setstacksize(&attributes, 8 * 1024 * 1024);
        if (!error) error = pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
        pthread_t thread;
        if (!error) error = pthread_create(&thread, &attributes, run_cli, nullptr);
        pthread_attr_destroy(&attributes);
    }
    if (error) {
        std::fprintf(stderr, "iOS host: Cannot create CLI thread (%d)\n", error);
        std::fflush(nullptr);
        _exit(1);
    }
}

@interface MoEDelegate : UIResponder <UIApplicationDelegate>
@property(nonatomic, strong) UIWindow* window;
@end
@implementation MoEDelegate
- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)options {
    (void)application; (void)options;
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    const bool command_mode = cli_argc > 1 || std::getenv("MOE_HOST_CLEAN_ONLY");
    UIViewController* view = command_mode ? [UIViewController new] : [MoEChatController new];
    view.view.backgroundColor = UIColor.systemBackgroundColor;
    self.window.rootViewController = view;
    [self.window makeKeyAndVisible];
    if (cli_argc > 1 || std::getenv("MOE_HOST_CLEAN_ONLY")) {
        start_cli();
    }
    return YES;
}
@end

int main(int argc, char** argv) {
    cli_argc = argc; cli_argv = argv;
    @autoreleasepool { return UIApplicationMain(argc, argv, nil, NSStringFromClass(MoEDelegate.class)); }
}
