// On-disk persistence of a conversation's state (SavedConversation): one file holding the running state, the
// checkpoints the caller passes (the engine passes only the deepest one), every QSA layer's authoritative K/V and
// the draft layer's K/V.  Pure host code: no CUDA.  Format v1 and the fail-closed read order are in
// docs/DETAILS.md (Session files).
//
// Format v1 is little-endian, with fixed-width integers and IEEE-754 floats; a big-endian build is refused at
// compile time.  The hashes detect accidental corruption; they do not authenticate a file: restore trusted files only.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

// What the file is bound to: the model files it was computed with, and the engine settings that change what the
// saved bytes mean.  A read with any other identity is refused before the payload is parsed.
struct SessionFileIdentity {
    uint64_t model = 0, config = 0;
};

// Non-cryptographic 64-bit hash with xxHash64-style rounds (NOT the standard XXH64 stream): corruption and identity
// checks, not adversarial input.  Streaming and one-shot agree.
class SessionHasher {
public:
    explicit SessionHasher(uint64_t seed = 0);
    void update(const void* data, size_t n);
    uint64_t digest() const;
private:
    void block(const uint8_t* p);
    uint64_t lane_[4];
    uint8_t pending_[32];
    size_t npending_ = 0;
    uint64_t total_ = 0;
};
uint64_t session_hash64(const void* data, size_t n, uint64_t seed);

// Typed, length-delimited fields for an identity hash: each field is (tag, type, length, bytes), so two different
// field lists never give the same byte stream, and a double is hashed by its exact bits (no decimal rounding).
class SessionIdentityBuilder {
public:
    explicit SessionIdentityBuilder(uint64_t seed) : hash_(seed) {}
    void str(const std::string& tag, const std::string& value);
    void i64(const std::string& tag, int64_t value);
    void u64(const std::string& tag, uint64_t value);
    void f64(const std::string& tag, double value);
    void bytes(const std::string& tag, const void* data, size_t n);
    uint64_t digest() const { return hash_.digest(); }
private:
    void field(const std::string& tag, uint8_t type, const void* data, uint64_t n);
    SessionHasher hash_;
};

// The RESOLVED rope configuration (rope_scaling.hpp, after the CLI flags and the model file's keys met).  The cached
// K is post-RoPE, so a different rotation makes the saved K mean something else.
struct SessionRope {
    int64_t type = 0;
    double freq_base = 0, factor = 0, freq_scale = 0, orig_ctx = 0, ext_factor = 0, attn_factor = 0, beta_fast = 0,
           beta_slow = 0;
};

// Every engine setting that changes what the saved state means.  Sampling, seeds, draft tuning, the expert tier and
// the prompt-reading chunk are not here: they change what is computed next, not what the saved cells hold.
struct SessionConfig {
    std::string engine_version;     // a file is bound to one engine version
    std::string backend;            // "cuda" or "hip": no cross-backend restore
    std::string kv;                 // --kv
    int64_t max_context = 0, kv_resident = 0, mtp_window = 0;
    bool kv_rot = false;            // STRATA_KV_ROT
    SessionRope rope;
    // the control vectors as loaded: a digest of the tables the engine uploaded (every file's content x its exact
    // scale, summed), the mode, the layer range and the direction; 0 when none is loaded
    uint64_t cvec = 0;
    // the arithmetic switches that change the computed state (name, value), in a fixed order
    std::vector<std::pair<std::string, int64_t>> switches;
};
uint64_t session_config_fingerprint(const SessionConfig& config);

// One model input the engine loaded and the role it plays ("native shard 1", "expert blk.3.ffn_up", ...).  The role,
// not the path, enters the fingerprint, so a moved model folder still matches.  Every (role, file) pair counts: one
// file in two roles is two entries (its bytes are read once).  `optional`: a missing file is recorded as absent
// instead of failing (a pack without experts.bin).
struct SessionModelFile {
    std::string role, path;
    bool optional = false;
};
// Sampled fingerprint of the model inputs: role, size, first and last MiB of each entry, in the given order (a file
// listed under several roles is read once).  Paths are UTF-8 (wide APIs on Windows; invalid UTF-8 is refused).  An
// edit in the middle of a file that keeps its size is NOT detected: model files must not change while sessions saved
// with them are kept.  `beat` (optional) is called after each distinct file is read.
bool session_model_fingerprint(const std::vector<SessionModelFile>& files, uint64_t& fingerprint, std::string& error,
                               const std::function<void()>& beat = {});

// What the engine loaded, as its loaders resolved it (generate.cpp fills it).  `experts`: (role, file) of the expert
// source - the pack's experts.bin, or every expert tensor read from a GGUF in place, per layer and role, as
// native_experts.txt resolved it (ExpertSource::model_inputs).
struct SessionInputs {
    std::vector<std::string> native_shards, native_dense, native_head;
    std::string embedding, ple, pack, mtp;
    std::vector<std::pair<std::string, std::string>> experts;
};
// The fingerprint's list: every role with its file, none merged (a path in two roles is two entries).
std::vector<SessionModelFile> session_model_inputs(const SessionInputs& inputs);

// Why a session file operation failed, for the caller's answer (the server maps it to an HTTP status):
//   invalid   refused: not a session file, corrupt, another model or configuration, over this engine's limits, a
//             link, nothing to save (400; nothing changed)
//   storage   no room on the disk: the free-space preflight, or ENOSPC / EDQUOT / EFBIG (507)
//   memory    the RAM preflight refused it, or an allocation failed (503)
//   io        any other failure: permissions, read/write/flush/rename errors, a missing folder (500)
enum class SessionError { none, invalid, storage, memory, io };
const char* session_error_name(SessionError e);
struct SessionStatus {
    SessionError error = SessionError::none;
    // write: the new file has replaced `path` (the rename happened).  Together with a failure: the file's bytes are
    // complete and flushed, but the folder flush after the rename failed, so the new name may not survive a power loss
    bool published = false;
    // write (POSIX): this filesystem cannot flush a folder (fsync EINVAL); the save succeeded without that flush
    bool dir_flush_unsupported = false;
};

// Progress of a file transfer: (bytes done, bytes in the file).  A write reports once when its temporary file is
// created (0) and after every block it hands to the OS (16 MiB, or less for the last one: a small file reports too);
// a read reports after every block it reads (16 MiB, or what is left; the first one, which holds the header, too).
// Each report follows a completed transfer of at most 16 MiB: no step between two reports moves more.
using SessionProgress = std::function<void(uint64_t done, uint64_t total)>;
// A step that does not stream - one call that blocks until it is done, over far more than 16 MiB or on the OS's
// schedule (flushing the file, the rename and folder flush) - is announced before it starts, with the bytes it
// concerns.  The engine turns this into an explicit, bounded allowance (session_phase_limit_s) for its watchdog and
// for the server; when the allowance runs out the step is stuck.
using SessionPhase = std::function<void(const char* phase, uint64_t bytes)>;
// The longest a non-streaming step over `bytes` may take before both sides call it stuck: 60 s plus one second per
// 4 MiB (a slow disk or network folder at 4 MiB/s), at most an hour.  1.2 GB: 345 s.
int64_t session_phase_limit_s(uint64_t bytes);

// Optional knobs of a write.
struct SessionWriteOptions {
    // the free-space PREFLIGHT: refused when the disk has less than the new file plus this many bytes free before the
    // write starts.  Not a reservation: other writers can take the space afterwards.
    uint64_t min_free_bytes = 0;
    bool durable = true;            // flush the file before the rename and (POSIX) the folder after it
    SessionProgress progress;
    // "flush" (the file's flush, `bytes` = the file) and "publish" (rename and folder flush), before each starts
    SessionPhase phase;
    // tests: return an errno value to make a step fail ("write", "file_flush", "rename", "dir_flush"), 0 to go on
    std::function<int(const char* step)> fault;
};

// Writes a temporary file beside `path` (a unique hidden name, created exclusively: never an existing file or link;
// mode 0600 on POSIX), flushes it, renames it over `path` and, on POSIX, flushes the folder (Windows has no folder
// flush: the rename is MoveFileExW with MOVEFILE_WRITE_THROUGH).  A failure BEFORE the rename keeps an existing file
// at `path` and removes only this call's own temporary file.  A failure of the folder flush AFTER the rename returns
// false with `status->published`: the new file is at `path` (the old one is gone), its directory entry may not
// survive a power loss.  A filesystem that cannot flush a folder (EINVAL) is not a failure; `status` says so.
bool session_file_write(const std::string& path, const SavedConversation& image, const SessionFileIdentity& id,
                        size_t& bytes, std::string& error, const SessionWriteOptions& options = {},
                        SessionStatus* status = nullptr);

// One K/V layer read straight from where it lives (device or pinned host pool) into the file's staging buffer,
// without a full host copy first.  `read(part, offset, dst, n)` copies bytes [offset, offset+n) of part
// 0..4 = k, v, k_scale, v_scale, pooled; after a failed read `error()` (optional) says why.  The file bytes are
// identical to writing a captured ConversationKv.
struct SessionKvSource {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<size_t, 5> sizes{};
    std::function<bool(size_t part, size_t offset, void* dst, size_t n)> read;
    std::function<std::string()> error;
};
// `meta` carries everything but the K/V (meta.kv must be empty); `kv` supplies the layers in order.
bool session_file_write(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                        const SessionFileIdentity& id, size_t& bytes, std::string& error,
                        const SessionWriteOptions& options = {}, SessionStatus* status = nullptr);

// The checkpoints worth a disk write: the deepest one (the next turn's resume point when the live tail was
// rewritten).  Earlier checkpoints only serve edits further back and cost ~118 MB each at this geometry.
std::vector<ConversationCheckpoint> session_checkpoints_to_save(const std::vector<ConversationCheckpoint>& chain);
// The same choice without a copy (nullptr for an empty chain).
const ConversationCheckpoint* session_deepest_checkpoint(const std::vector<ConversationCheckpoint>& chain);

// What a SAVE copies into RAM besides what the engine already holds, for its RAM preflight.
struct SessionSaveLive {
    uint64_t state_bytes = 0;   // the live running state (gdn + ple + every owned QSA layer's tail, dead, block_pos)
    uint64_t tokens = 0, images = 0;
    uint64_t kv_layers = 0;     // K/V sources (owned QSA layers + the draft)
};
// The peak, with saturating arithmetic: two copies of the deepest checkpoint (the list handed to the capture and the
// one inside the captured image), the copied live state with its token and image lists, the K/V source directory,
// the 16 MiB write buffer and 1 MiB for the small vectors.
uint64_t session_save_peak_bytes(const ConversationCheckpoint* deepest, const SessionSaveLive& live);
// The SAVE's RAM PREFLIGHT, BEFORE anything is copied: `admit` is asked for session_save_peak_bytes; only when it
// agrees is the deepest checkpoint copied into `out`.  On a refusal nothing was allocated, `out` is empty, `chain`
// and the live state are untouched and no file was opened.  Not a reservation.
bool session_save_checkpoints(const std::vector<ConversationCheckpoint>& chain, const SessionSaveLive& live,
                              const std::function<bool(uint64_t need_bytes, std::string& why)>& admit,
                              std::vector<ConversationCheckpoint>& out, std::string& why);

// What a read checks before it allocates.  Every count is checked against the bytes left in the payload and against
// these limits BEFORE its array is allocated; the defaults accept whatever the payload can hold.  The engine sets all
// of them from its runtime (conversation_session_read_limits), and then the file size is bounded too.
struct SessionReadLimits {
    uint64_t max_file_bytes = UINT64_MAX;
    uint64_t max_tokens = UINT64_MAX;       // token and image counts of the live state and of each checkpoint
    uint64_t max_checkpoints = UINT64_MAX;
    uint64_t max_kv_layers = UINT64_MAX;
    // the runtime's geometry and layer range: a file with others is refused before its state arrays are read
    std::optional<std::array<int64_t, 18>> geometry;
    std::optional<std::pair<int64_t, int64_t>> layer_range;
    // the most bytes of each running-state array (gdn, ple, tails, dead, block_pos), live and per checkpoint
    std::array<uint64_t, 5> max_state_bytes{UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    // per K/V layer (the QSA layers, then the draft): the most bytes of each part (k, v, k_scale, v_scale, pooled).
    // A layer past the end of the list is bounded by the payload only.
    std::vector<std::array<uint64_t, 5>> max_kv_bytes;
    // the RAM PREFLIGHT: asked once the header and identity match and before any payload allocation, with what the
    // parse holds at its peak (session_read_peak_bytes); false refuses the file.  Not a reservation.
    std::function<bool(uint64_t need_bytes, std::string& why)> admit;
    SessionProgress progress;
};
// The largest file these limits admit (UINT64_MAX when one of them is open); session_file_read applies it.
uint64_t session_read_max_file_bytes(const SessionReadLimits& limits);
// What the RAM preflight is asked for a file of `file_bytes`: the parsed image (about the payload), the 16 MiB read
// buffer, and the per-allocation overhead of its arrays (a page per 16 MiB segment plus the small vectors).
uint64_t session_read_peak_bytes(uint64_t file_bytes);
// Full validation (a regular file with one link, size, header, identity, bounded parse, payload hash) before `image`
// is replaced.  The file is opened without following a symbolic link (or a reparse point) at its name, and the checks
// are made on the opened handle.  On failure `image` is unchanged and `status->error` says what kind of failure.
bool session_file_read(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
                       size_t& bytes, std::string& error, const SessionReadLimits& limits = {},
                       SessionStatus* status = nullptr);

// The folder the free-space preflight asks about for `path`, by the rules of Windows or of POSIX (exposed so both can
// be tested anywhere).  Windows: backslashes and a trailing backslash, as GetDiskFreeSpaceExW requires for a UNC
// folder (\\server\share\dir\) and gives a drive root (C:\); empty for a bare file name (the current folder).
std::string session_free_space_dir(const std::string& path, bool windows_rules);

} // namespace strata::core
