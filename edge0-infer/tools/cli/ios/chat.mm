#include "chat.h"
#include "edge0/diagnostics.h"
#include "edge0/generation.h"
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>

namespace {
template<class T> T checked(edge0::Result<T> result) {
    if (!result) throw std::runtime_error(result.error().message);
    return result.value();
}
NSString* text(const std::string& value) {
    NSString* result = [[NSString alloc] initWithBytes:value.data() length:value.size() encoding:NSUTF8StringEncoding];
    return result ? result : @"Invalid UTF-8";
}
struct ChatState {
    std::optional<edge0::RuntimeReference> runtime;
    edge0::ModelReference model;
    edge0::SessionReference session;
    edge0::GenerationConfiguration configuration;
    std::vector<edge0::TextMessage> history;
    void close() {
        if (session) edge0::release_session(session);
        if (model) edge0::unload_model(model);
        if (runtime) edge0::finish_runtime_use(*runtime);
        session.reset(); model.reset(); runtime.reset();
    }
};
}

@implementation MoEChatController {
    dispatch_queue_t _work;
    std::shared_ptr<ChatState> _state;
    // Only the main thread reads/writes this handle; generation owns a copy.
    std::optional<edge0::CancelToken> _cancel;
    UITextView* _messages;
    UITextView* _input;
    UILabel* _status;
    UIButton* _send;
    UIButton* _stop;
    UIButton* _reset;
}
- (void)dealloc {
    auto state = _state;
    if (_work) dispatch_async(_work, ^{ state->close(); });
}
- (void)viewDidLoad {
    [super viewDidLoad];
    _state = std::make_shared<ChatState>();
    _work = dispatch_queue_create("dev.edge0.chat.worker", DISPATCH_QUEUE_SERIAL);
    self.view.backgroundColor = UIColor.systemBackgroundColor;
    NSString* modelPath = [NSBundle.mainBundle objectForInfoDictionaryKey:@"MoEModelDirectory"];
    _status = [UILabel new]; _status.numberOfLines = 0;
    _status.text = [NSString stringWithFormat:@"Loading Documents/%@...", modelPath];
    _messages = [UITextView new]; _messages.editable = NO;
    _messages.font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
    _input = [UITextView new]; _input.font = _messages.font;
    _input.backgroundColor = UIColor.secondarySystemBackgroundColor;
    _input.accessibilityLabel = @"Message";
    _send = [UIButton buttonWithType:UIButtonTypeSystem]; [_send setTitle:@"Send" forState:UIControlStateNormal];
    _stop = [UIButton buttonWithType:UIButtonTypeSystem]; [_stop setTitle:@"Stop" forState:UIControlStateNormal];
    _reset = [UIButton buttonWithType:UIButtonTypeSystem]; [_reset setTitle:@"New conversation" forState:UIControlStateNormal];
    [_send addTarget:self action:@selector(sendMessage) forControlEvents:UIControlEventTouchUpInside];
    [_stop addTarget:self action:@selector(stopMessage) forControlEvents:UIControlEventTouchUpInside];
    [_reset addTarget:self action:@selector(resetConversation) forControlEvents:UIControlEventTouchUpInside];
    UIStackView* buttons = [[UIStackView alloc] initWithArrangedSubviews:@[_send, _stop, _reset]];
    buttons.distribution = UIStackViewDistributionEqualSpacing;
    UIStackView* column = [[UIStackView alloc] initWithArrangedSubviews:@[_status, _messages, _input, buttons]];
    column.axis = UILayoutConstraintAxisVertical; column.spacing = 8;
    column.translatesAutoresizingMaskIntoConstraints = NO; [self.view addSubview:column];
    [NSLayoutConstraint activateConstraints:@[
        [column.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor constant:12],
        [column.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor constant:12],
        [column.trailingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor constant:-12],
        [column.bottomAnchor constraintEqualToAnchor:self.view.keyboardLayoutGuide.topAnchor constant:-8],
        [_input.heightAnchor constraintEqualToConstant:80]]];
    [self ready:NO];
    auto state = _state;
    dispatch_async(_work, ^{
        @autoreleasepool {
            try {
                std::string device;
                for (const auto& item : edge0::query_capabilities().devices)
                    if (item.backend == "metal") { device = item.name; break; }
                if (device.empty()) throw std::runtime_error("Metal device unavailable: no compiled Metal backend device found");
                edge0::RuntimeOptions options; options.devices = {device};
                state->runtime = checked(edge0::open_runtime(options));
                NSString* documents = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
                if (!modelPath.length) throw std::runtime_error("Missing App model directory configuration");
                std::string directory = std::string(documents.UTF8String) + "/" + modelPath.UTF8String;
                edge0::LoadOverrides loading; loading.device = device;
                auto resolved = checked(edge0::resolve_load_configuration(*state->runtime, directory, loading));
                state->model = checked(edge0::load_model(*state->runtime, resolved));
                state->session = checked(edge0::create_session(state->model));
                edge0::GenerationOverrides generation; generation.thinking = false;
                state->configuration = checked(edge0::resolve_generation_configuration(state->model, generation));
                dispatch_async(dispatch_get_main_queue(), ^{
                    self->_status.text = @"Ready"; [self ready:YES];
                    std::fprintf(stderr, "chat ui: model loaded\n"); std::fflush(stderr);
                });
            } catch (const std::exception& error) {
                std::fprintf(stderr, "chat ui: load failed: %s\n", error.what()); std::fflush(stderr);
                NSString* message = [NSString stringWithFormat:@"Cannot load Documents/%@: %@", modelPath, text(error.what())];
                dispatch_async(dispatch_get_main_queue(), ^{ self->_status.text = message; });
            }
        }
    });
}
- (void)ready:(BOOL)ready {
    _send.enabled = ready; _reset.enabled = ready; _input.editable = ready; _stop.enabled = NO;
}
- (void)append:(NSString*)value {
    _messages.text = [_messages.text stringByAppendingString:value];
    [_messages scrollRangeToVisible:NSMakeRange(_messages.text.length, 0)];
}
- (void)sendMessage {
    NSString* input = _input.text;
    if (!input.length || !_send.enabled) return;
    [self append:[NSString stringWithFormat:@"\nYou: %@\nAssistant: ", input]];
    _input.text = @""; [self ready:NO]; _status.text = @"Generating...";
    const std::string prompt(input.UTF8String);
    auto state = _state;
    dispatch_async(_work, ^{
        @autoreleasepool {
            try {
                auto token = checked(edge0::create_cancel_token(*state->runtime));
                // Publish the token before generation; Stop is a main-thread atomic request.
                dispatch_sync(dispatch_get_main_queue(), ^{ self->_cancel = token; self->_stop.enabled = YES; });
                auto candidate = state->history; candidate.push_back({"user", prompt});
                auto receiver = [self](const edge0::OutputEvent& event) -> std::optional<edge0::Error> {
                    if (event.kind == edge0::OutputKind::answer) {
                        NSString* fragment = text(event.text);
                        dispatch_async(dispatch_get_main_queue(), ^{ [self append:fragment]; });
                    }
                    return std::nullopt;
                };
                const auto result = checked(edge0::generate(state->session, candidate, state->configuration, {}, receiver, &token));
                const bool normal = result.stop_reason == edge0::StopReason::length ||
                    result.stop_reason == edge0::StopReason::end_of_generation || result.stop_reason == edge0::StopReason::stop_string;
                if (result.state_valid && !result.error && (normal ||
                    (result.stop_reason == edge0::StopReason::cancelled && !result.text.empty()))) {
                    state->history = candidate; state->history.push_back({"assistant", result.text});
                }
                const auto memory = edge0::memory_snapshot(*state->runtime);
                NSString* first = result.request_summary.first_token_seconds ?
                    [NSString stringWithFormat:@"%.2f s", *result.request_summary.first_token_seconds] : @"unavailable";
                NSString* speed = result.request_summary.decode_tokens_per_second ?
                    [NSString stringWithFormat:@"%.2f tok/s", *result.request_summary.decode_tokens_per_second] : @"unavailable";
                NSString* peak = @"unavailable";
                if (memory && memory.value().physical.values && memory.value().physical.values->peak_bytes)
                    peak = [NSString stringWithFormat:@"%.2f GiB", static_cast<double>(*memory.value().physical.values->peak_bytes) / (1024*1024*1024)];
                NSString* metrics = [NSString stringWithFormat:@"\nFirst token %@ | Decode %@ | Process peak %@\n", first, speed, peak];
                NSString* status = result.error ? text(result.error->message) : @"Ready";
                dispatch_async(dispatch_get_main_queue(), ^{
                    self->_cancel.reset(); [self append:metrics]; self->_status.text = status; [self ready:YES];
                });
            } catch (const std::exception& error) {
                NSString* message = text(error.what());
                dispatch_async(dispatch_get_main_queue(), ^{
                    self->_cancel.reset(); self->_status.text = message; [self ready:YES];
                });
            }
        }
    });
}
- (void)stopMessage {
    if (_cancel) { edge0::request_cancel(*_cancel); _stop.enabled = NO; _status.text = @"Stopping..."; }
}
- (void)resetConversation {
    [self ready:NO]; auto state = _state;
    dispatch_async(_work, ^{
        try {
            checked(edge0::release_session(state->session)); state->session.reset();
            state->session = checked(edge0::create_session(state->model)); state->history.clear();
            dispatch_async(dispatch_get_main_queue(), ^{ self->_messages.text = @""; self->_status.text = @"Ready"; [self ready:YES]; });
        } catch (const std::exception& error) {
            NSString* message = text(error.what());
            dispatch_async(dispatch_get_main_queue(), ^{ self->_status.text = message; [self ready:YES]; });
        }
    });
}
@end
