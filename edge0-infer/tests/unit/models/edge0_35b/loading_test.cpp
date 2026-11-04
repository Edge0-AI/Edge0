#include "model/loading.h"
#include "models/edge0_35b/assets.h"
#include "models/edge0_35b/compute.h"
#include "tests/support/runtime.h"
#include "tests/support/model_fixture.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
using namespace edge0;
using test::take;
int main(int argc, char** argv) {
    auto backend = test::backend(argc, argv);
    exec::Budget budget(1024ULL * 1024 * 1024);
    model::WeightDescription small;
    small.resident.push_back({"x", "base", "x", {"", 0, 60, {GGML_TYPE_F32, {3, 5}}}});
    small.resident.push_back({"y", "base", "y", {"", 0, 32, {GGML_TYPE_F16, {16}}}});
    const auto estimate = take(model::estimate_weights(small, backend));
    {
        weights::Weights weights(budget);
        auto storage = take(weights.create_storage(backend, {{{GGML_TYPE_F32, {3, 5}}, 1}, {{GGML_TYPE_F16, {16}}, 1}}, edge0::exec::Purpose::resident_weights));
        CHECK(budget.current() == estimate.total_bytes);
        CHECK(!weights.release_storage(storage));
    }
    CHECK(budget.current() == 0);
    {
        // Resident packing: each part stays within the buffer limit, order and ids are kept,
        // and a single tensor above the limit is refused.
        auto* type = ggml_backend_get_default_buffer_type(backend->handle());
        const auto alignment = ggml_backend_buft_get_alignment(type);
        const model::ResidentWeight a{"a", "base", "a", {"", 0, 1024, {GGML_TYPE_F32, {256}}}},
            b{"b", "base", "b", {"", 1024, 1024, {GGML_TYPE_F32, {256}}}},
            c{"c", "base", "c", {"", 2048, 1024, {GGML_TYPE_F32, {256}}}},
            large{"large", "base", "large", {"", 0, 4096, {GGML_TYPE_F32, {1024}}}};
        const auto one = take(model::placed_bytes({type, std::numeric_limits<std::size_t>::max(), alignment}, a.source.layout));
        CHECK(one >= 1024 && one % alignment == 0);
        const model::BufferBound two{type, 2 * one, alignment};
        auto parts = take(model::split_resident_group({&a, &b, &c}, two));
        CHECK(parts.size() == 2 && parts[0].size() == 2 && parts[1].size() == 1);
        CHECK(parts[0][0]->id == "a" && parts[0][1]->id == "b" && parts[1][0]->id == "c");
        CHECK(take(model::split_resident_group({&a, &b, &c}, {type, 3 * one, alignment})).size() == 1);
        CHECK(take(model::split_resident_group({&a, &b, &c}, model::buffer_bound({backend}))).size() == 1);
        auto refused = model::split_resident_group({&a, &large}, two);
        CHECK(!refused && refused.error().category == ErrorCategory::invalid_argument);
    }
    if (argc > 1 && std::string(argv[1]) == "--arithmetic") return 0;
    const auto* reference = std::getenv("MOE_TINY");
    if (!reference || !std::filesystem::exists(std::filesystem::path(reference) / "tiny-a/assets.tsv")) {
        std::cout << "SKIP: missing MOE_TINY tiny-a\n"; return 77;
    }
    const auto& adapter = models::edge0_35b::adapter();
    model::LoadRequest request;
    request.directory = (std::filesystem::path(reference) / "tiny-a").string();
    request.device = "CPU"; request.options["experts"]="resident";
    const std::map<std::string, std::shared_ptr<exec::Backend>> backends{{"CPU", backend}};
    model::LoadReport report;
    request.report = &report;
    int closes = 0;
    request.on_close = [&](model::LoadedModel& model) { models::edge0_35b::close_model(model); ++closes; };
    std::size_t required = 0;
    {
        auto result = model::load(request, {&adapter}, budget, backends);
        CHECK(result);
        auto loaded = result.value();
        required = loaded->estimate.total_bytes;
        CHECK(budget.current() == required && loaded->resident.size() == 273);
        CHECK(loaded->weights.read_count() == 273 && report.weight_reads == 273);
        for (const auto& item : loaded->resident) {
            const auto separator = item.first.find(':');
            const auto& tensor = loaded->assets.indexes.at(item.first.substr(0, separator)).tensors.at(item.first.substr(separator + 1));
            std::vector<char> expected(tensor.bytes), actual(tensor.bytes);
            std::ifstream input(tensor.path, std::ios::binary);
            input.seekg(static_cast<std::streamoff>(tensor.offset));
            input.read(expected.data(), static_cast<std::streamsize>(expected.size()));
            CHECK(input.good());
            ggml_backend_tensor_get(item.second.storage->tensor(item.second.binding.region), actual.data(), 0, actual.size());
            CHECK(expected == actual);
        }
        auto declarations = take(models::edge0_35b::describe_states(loaded->assets.definition));
        auto kv = take(session::KVCache::create(budget, backend, 1024, declarations.kv));
        session::StateStore states;
        const auto before_states = budget.current();
        CHECK(!models::edge0_35b::initialize_session(loaded, *kv, states, budget, exec::Stages{backend, backend}));
        const auto all_state_bytes = budget.current() - before_states;
        for (auto layer : loaded->assets.definition.layer_groups.at("deltanet")) {
            auto state = take(states.get<models::edge0_35b::DeltaState>("deltanet." + std::to_string(layer)));
            CHECK(state);
            CHECK(state->convolution->tensor()->ne[0] == 3 && state->convolution->tensor()->ne[1] == 1024);
            CHECK(state->recurrent->tensor()->ne[0] == 128 && state->recurrent->tensor()->ne[1] == 128 && state->recurrent->tensor()->ne[2] == 4);
            for (const auto& storage : {state->convolution, state->recurrent}) {
                std::vector<float> values(ggml_nelements(storage->tensor()));
                ggml_backend_tensor_get(storage->tensor(), values.data(), 0, values.size() * sizeof(float));
                CHECK(std::all_of(values.begin(), values.end(), [](float value) { return value == 0; }));
            }
        }
        models::edge0_35b::release_session(loaded, *kv, states);
        CHECK(budget.current() == before_states);
        models::edge0_35b::release_session(loaded, *kv, states);
        CHECK(budget.current() == before_states);
        for (auto layer : loaded->assets.definition.layer_groups.at("deltanet"))
            CHECK(!take(states.get<models::edge0_35b::DeltaState>("deltanet." + std::to_string(layer))));
        exec::Budget partial_budget(all_state_bytes / 2);
        auto partial = models::edge0_35b::initialize_session(loaded, *kv, states, partial_budget, exec::Stages{backend, backend});
        CHECK(partial && partial->category == ErrorCategory::budget_exceeded);
        CHECK(take(states.get<models::edge0_35b::DeltaState>("deltanet.0")) && partial_budget.current() > 0);
        models::edge0_35b::release_session(loaded, *kv, states);
        CHECK(partial_budget.current() == 0);
        auto session_hold = loaded;
        std::weak_ptr<weights::Storage> retained = loaded->storages.front();
        CHECK(!model::request_unload(loaded));
        CHECK(!loaded && budget.current() == before_states && closes == 0);
        CHECK(!retained.expired());
        auto initialize = models::edge0_35b::initialize_session(session_hold, *kv, states, budget, exec::Stages{backend, backend});
        CHECK(initialize && initialize->category == ErrorCategory::conflict);
        CHECK(budget.current() == before_states && !take(states.get<models::edge0_35b::DeltaState>("deltanet.0")));
        auto error = session_hold->require_active();
        CHECK(error && error->category == ErrorCategory::conflict);
    }
    CHECK(budget.current() == 0 && closes == 1);
    const auto alignment = ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend->handle()));
    exec::Budget insufficient(required - alignment);
    std::atomic<std::size_t> reads{0};
    request.before_read = [&](std::size_t) { ++reads; };
    {
        auto rejected = model::load(request, {&adapter}, insufficient, backends);
        CHECK(!rejected && rejected.error().category == ErrorCategory::budget_exceeded);
        CHECK(insufficient.current() == 0 && reads == 0 && report.weight_reads == 0);
    }
    request.device = "missing";
    auto absent = model::load(request, {&adapter}, budget, backends);
    CHECK(!absent && absent.error().category == ErrorCategory::unsupported && reads == 0);
    request.device = "CPU"; request.options["routing"] = "predicted"; request.options["lora"] = "off";
    auto predicted = model::load(request, {&adapter}, budget, backends);
    CHECK(!predicted && predicted.error().category == ErrorCategory::invalid_argument && reads == 0);
    CHECK(report.weight_reads == 0);
    request.options={{"experts","resident"}};
    {
        fixture::Temp missing;
        std::vector<std::pair<std::string, std::string>> files;
        auto catalog = take(model::open_catalog(request.directory));
        for (const auto& item : catalog.files) {
            if (item.first == "lora") continue;
            std::filesystem::create_symlink(item.second.path, missing.path / item.second.filename);
            files.emplace_back(item.first, item.second.filename);
        }
        fixture::catalog(missing.path, files);
        auto no_lora = request; no_lora.directory = missing.path.string();
        auto rejected = model::load(no_lora, {&adapter}, budget, backends);
        CHECK(!rejected && rejected.error().category == ErrorCategory::incompatible);
        CHECK(report.weight_reads == 0 && budget.current() == 0);
    }
    {
        auto mixed_request=request;mixed_request.before_read={};mixed_request.on_close={};
        mixed_request.prefill_device="other";
        auto facility=platform::BackendFacility::create("CPU");CHECK(facility.first);
        auto owner=std::make_shared<platform::BackendFacility>(std::move(facility.first));
        auto other=take(exec::resolve_stages("CPU","CPU",owner)).prefill;
        auto pair=backends;pair.emplace("other",other);
        auto rejected=model::load(mixed_request,{&adapter},budget,pair);
        CHECK(!rejected && rejected.error().category==ErrorCategory::unsupported);
        CHECK(report.weight_reads==0 && budget.current()==0);
        mixed_request.supports_handoff=true;
        mixed_request.consumption=[](auto,auto){return exec::Consumption::copy;};
        exec::Budget limited(required);
        rejected=model::load(mixed_request,{&adapter},limited,pair);
        CHECK(!rejected && rejected.error().category==ErrorCategory::budget_exceeded);
        CHECK(report.weight_reads==0 && limited.current()==0 && limited.peak()==0);
        {
            // Storage alone fits; the 16 MiB write staging must be refused before any read.
            std::size_t mixed_required=0;
            { auto loaded=take(model::load(mixed_request,{&adapter},budget,pair)); mixed_required=loaded->estimate.total_bytes; }
            CHECK(budget.current()==0);
            { exec::Budget exact(mixed_required); CHECK(model::load(mixed_request,{&adapter},exact,pair)); }
            mixed_request.direct=[](auto,auto){return false;};
            exec::Budget staging_short(mixed_required);
            rejected=model::load(mixed_request,{&adapter},staging_short,pair);
            CHECK(!rejected && rejected.error().category==ErrorCategory::budget_exceeded);
            CHECK(report.weight_reads==0 && staging_short.current()==0 && staging_short.peak()==0);
        }
        {
            // Only the admission probe may see staging unreserved; every write after it must find it held.
            int unreserved=0,reserved=0;
            mixed_request.direct=[&](auto,auto){(budget.current(edge0::exec::Purpose::staging)==0?unreserved:reserved)++;return false;};
            const auto reads_before=report.weight_reads;
            auto staged=take(model::load(mixed_request,{&adapter},budget,pair));
            CHECK(staged->stages.mixed() && budget.current(edge0::exec::Purpose::staging)==4*std::size_t(4*1024*1024));
            CHECK(report.weight_reads>reads_before);
            CHECK(unreserved==1 && reserved>0);
        }
        CHECK(budget.current()==0);
        mixed_request.direct={};
        for(bool copy:{false,true}) {
            mixed_request.consumption=copy?exec::ConsumptionQuery([](auto,auto){return exec::Consumption::copy;}):exec::ConsumptionQuery{};
            auto loaded=take(model::load(mixed_request,{&adapter},budget,pair));
            CHECK(loaded->estimate.total_bytes==budget.current());
            CHECK(loaded->stages.mixed());
        }
        CHECK(budget.current()==0);
    }
    {
        // Single-backend path: same staging admission when the inject says the buffer is not writable.
        auto staged_request=request;staged_request.before_read={};staged_request.on_close={};
        staged_request.direct=[](auto,auto){return false;};
        exec::Budget staging_short(required);
        auto rejected=model::load(staged_request,{&adapter},staging_short,backends);
        CHECK(!rejected && rejected.error().category==ErrorCategory::budget_exceeded);
        CHECK(report.weight_reads==0 && staging_short.current()==0 && staging_short.peak()==0);
        int unreserved=0,reserved=0;
        staged_request.direct=[&](auto,auto){(budget.current(edge0::exec::Purpose::staging)==0?unreserved:reserved)++;return false;};
        auto staged=take(model::load(staged_request,{&adapter},budget,backends));
        CHECK(budget.current(edge0::exec::Purpose::staging)==4*std::size_t(4*1024*1024));
        CHECK(unreserved==1 && reserved>0);
        CHECK(staged->estimate.total_bytes==budget.current());
    }
    CHECK(budget.current()==0);
    request.before_read = [](std::size_t region) { if (region == 0) throw std::runtime_error("Injected read failure"); };
    auto failed = model::load(request, {&adapter}, budget, backends);
    CHECK(!failed && failed.error().category == ErrorCategory::backend_error);
    CHECK(budget.current() == 0);
    std::cout << "load: resident byte comparison, admission, rollback and deferred unload passed; required=" << required << '\n';
}
