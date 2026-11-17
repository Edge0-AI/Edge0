#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

#include <string>

namespace strata::core {

// Caller synchronizes the device before saving, and after restoring all layers.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
// A nonzero unchanged_tokens is valid only for storage retained from an image
// actually restored into this session, bounded by every subsequent rewrite.
// Equal token IDs alone do not establish that its K/V bytes are unchanged.
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error,
                          int64_t unchanged_tokens = 0, size_t* reused_bytes = nullptr);
bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                   int64_t upto, bool include_index, size_t& bytes, std::string& error);
// No CUDA calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error);
// conversation_kv_save(from) + conversation_kv_restore(to) without a host image, for two states of the same layout.
// The copies are complete on return; the residency refresh is queued on the legacy stream like the restore's.
// `queued`: the copies are only queued on the private copy stream, the refresh is ordered after them on the device,
// and the caller synchronizes the device before it reads or writes either state.
bool conversation_kv_copy(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error, bool queued = false);
// The first `upto` tokens of an image captured at `captured` tokens: the leading bytes of each buffer (cells come
// first in the identity layout, as conversation_kv_save takes them), the image untouched.  Cells and indexer rows
// past `upto` in the restored pages hold the image's later tokens until a request writes them, as after a whole
// restore followed by a rewind to a checkpoint.
bool conversation_kv_restore_prefix(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                    int64_t captured, int64_t upto, bool include_index, std::string& error);
// Diagnostic read-back after a synchronized restore. Uses 64 KiB of stack
// workspace, compares authoritative bytes and resident draft-ring pages, and
// fingerprints the authoritative payload only. Never changes model state.
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, uint64_t& fingerprint, std::string& error);
// Disk save without a host copy: the layer's authoritative K/V as a streamed source (same bytes and metadata as
// conversation_kv_save).  Caller synchronizes the device first and keeps the state untouched while it is read.
bool conversation_kv_source(SessionKvSource& source, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, std::string& error);
// The bytes of each K/V part (k, v, k_scale, v_scale, pooled) a snapshot of `upto` tokens holds: no state read.
bool conversation_kv_part_sizes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                                std::array<uint64_t, 5>& sizes, std::string& error);
// Disk sessions: the read limits this session can ever restore - its geometry and layer range, at most
// min(max_tokens, its cells) tokens, `max_checkpoints` checkpoints, the exact running-state sizes and the K/V part
// sizes at that many tokens - so session_file_read refuses an oversized or foreign file before it allocates.
bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& session, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
/// Whole-model sizes: `gdn` covers every GDN layer, the indexer sizes are per QSA layer.
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
/// The same for one session's layer carve (#216): `gdn` covers its `gdn_alloc` rows; the per-QSA-layer sizes
/// apply to each owned state [qsa_ord0, qsa_ord0 + qsa_alloc).  Rejects an inconsistent carve.  Checkpoints,
/// snapshots and their validation all use this: a session saves and restores only the state it owns.
bool conversation_session_sizes(const ModelGeometry& g, const SessionState& session, ConversationStateSizes& sizes,
                                std::string& error);
bool conversation_checkpoint_validate(const ConversationCheckpoint& checkpoint, const SessionState& session,
                                      const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_save(ConversationCheckpoint& checkpoint, const SessionState& session,
                                  const ModelGeometry& g, std::string& error);
/// STRATA_CKPT_ASYNC=1 (CUDA, opt-in): conversation_checkpoint_save copies the GDN state into a device staging buffer
/// and moves it into the checkpoint's host vector on a background thread (the PCIe copy overlaps what follows: the
/// rest of the prompt, the reply).  This waits for that copy; every reader of a checkpoint's bytes calls it first
/// (restore, snapshot save / restore, the parked copies, the session file, the serve loop at each request).  No-op
/// when nothing is in flight.
/// `only` (non-null): wait only when the copy in flight is into that checkpoint's GDN bytes.
void conversation_checkpoint_join(const void* only = nullptr);
/// STRATA_PARK_ASYNC: waits until the last parked image's K/V has landed in its host buffers (no-op otherwise); the
/// readers of a parked image's K/V call it first, and the image itself before it frees them (SavedConversation::pending).
void conversation_park_join();
/// STRATA_PARK_ASYNC=1 (CUDA, opt-in): between begin and launch, conversation_kv_save copies the device K/V device to
/// device into a VRAM stage (only while 1.5 GB of VRAM stays free beside it; else as before) and records where it
/// goes; launch moves the stage into the host buffers on a background thread (4 MiB pieces through two pinned
/// buffers), so the next prompt starts at once.  conversation_park_join (above) waits for it.
/// begin: false when it is off (the saves then copy as before); `need` is the K/V bytes the saves will copy.
bool conversation_park_async_begin(size_t need);
/// The recorded copies start (the device copies into the stage have landed when it returns); false: a copy failed.
bool conversation_park_async_launch(std::string& error);
/// Abandons a begun park that was not launched (the error paths): nothing was queued to the background.
void conversation_park_async_abandon();
/// STRATA_PARK_VRAM_MIB=N (with STRATA_PARK_ASYNC): a park also keeps the whole K/V of its image in VRAM, in a slot
/// of its own taken at begin while the slots' total stays within N MiB (and 1.5 GB of VRAM stays free), so a restore
/// of that image copies device to device from the slot (conversation_kv_restore / _prefix) instead of host to
/// device, and does not wait for the background copy.  The slot lives as long as the image and its K/V layers
/// (SavedConversation::pending, ConversationKv::pending).  Default 0: off.
/// The tag of the begun park's slot (0: none); kept until the next begin, for the tokens.
uint64_t conversation_park_slot_tag();
/// Frees the slot under `tag` (no-op for 0 or a slot already gone); the caller joined the copy that reads it.
void conversation_park_slot_release(uint64_t tag);
/// VRAM pressure: joins the background copy, then frees every slot and the stage.  Returns the bytes freed (0, and
/// nothing is touched, while STRATA_PARK_VRAM_MIB is unset).
size_t conversation_park_vram_release_all();
/// conversation_park_join, unless every K/V layer of `image` has its slot (the restore then reads no host buffer
/// the background copy could still be filling; a layer that falls back to the host joins for itself).
void conversation_park_join_for(const SavedConversation& image);
bool conversation_checkpoint_restore(const ConversationCheckpoint& checkpoint, SessionState& session,
                                     const ModelGeometry& g, std::string& error);
// conversation_checkpoint_save(from) + conversation_checkpoint_restore(to) of `ids` without a host image, for two
// sessions carved alike on the same device; the caller synchronizes the device before (as for the restore).
// `queued`: the copies are only queued (the private copy stream), and the caller synchronizes the device after too.
bool conversation_checkpoint_copy(const SessionState& from, SessionState& to, const std::vector<int32_t>& ids,
                                  const ModelGeometry& g, std::string& error, bool queued = false);

struct ConversationView {
    const std::vector<int32_t>& ids;
    const std::vector<ConversationImageKey>& images;
    const std::vector<ConversationCheckpoint>& checkpoints;
    bool cvec;
};
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error);
// The capture estimate includes retained capacity and transient segment directories;
// only estimate - reuse.bytes() requires additional physical RAM. Capture consumes
// the uniquely owned reusable buffers, including on failure.
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
// Disk save without capturing the K/V on the host: `meta` gets everything but the K/V (running state copied,
// checkpoints as given by the view), `sources` one streamed source per QSA layer then the draft.  Caller has
// synchronized and must not run the session until the file is written.
bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& session,
                                   const ModelGeometry& g, const QsaState& draft, std::string& error);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
/// #2409: true when a restore (or a copy) on CUDA device `device` from the calling thread creates no copy stream: this
/// thread has one for the card or a finished thread left one in the pool. A restore writes into buffers that exist
/// already; a new stream is the one thing it can allocate (see program/conv_cache.hpp restore_fits).
bool conversation_copy_stream_ready(int device);
enum class ConversationRestore { restored, invalid, transfer_failed };
// Invalid images are rejected before any CUDA call/write. Transfer failure may
// leave partial state: caller MUST NOT continue inference from that session.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);
// The K/V of the image's first `upto` tokens (main layers and the draft layer), not its running state: the caller
// mounts a checkpoint taken at `upto`.  For a checkpoint of a parked conversation that a new conversation starts
// with, while the parked one stays where it is (ConversationCache::borrows).  Same failure contract as above.
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& session, const ModelGeometry& g,
                                                          const QsaState& draft, std::string& error);

// The same with `draft == nullptr`: an image WITHOUT the draft layer's K/V (kv holds the session's own QSA layers
// only).  A layer split's later stages park this way; the draft ring is saved once, with the first stage's image.
// An image is validated and restored with the same kind of call it was saved with (a K/V layer count mismatch is
// rejected as invalid).
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session, const ModelGeometry& g,
                                 const QsaState* draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                         size_t& bytes, std::string& error);
// `storage` (optional, STRATA_CKPT_POOL): dropped checkpoints whose vectors the image's running state and checkpoint
// copies go into (their pages are in: no fresh pages to fault); the ones not used are left in it.
// `moved` (optional): checkpoints the caller drops after this, moved into the image after the copies of the view's
// (the image holds the same checkpoints as with all of them in the view; nothing is copied for these)
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& session,
                                const ModelGeometry& g, const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr,
                                std::vector<ConversationCheckpoint>* storage = nullptr,
                                std::vector<ConversationCheckpoint>* moved = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error);
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error);
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& session, const ModelGeometry& g,
                                                          const QsaState* draft, std::string& error);

} // namespace strata::core
