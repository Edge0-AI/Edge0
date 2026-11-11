"""tools/mtp_fetch.py - plan v0.3, P0.3/P6: the MTP block from the BF16 checkpoint, without the checkpoint.

The GSQ-RCO GGUF ships no MTP head. The BF16 checkpoint (Qwen/Qwen3.8-Flash-Next, 360 GB in 131 shards) does:
31 `mtp.*` tensors scattered over 28 shards. Safetensors puts a JSON header (name -> dtype, shape, byte range)
at the start of each shard, so HTTP range requests can read the headers and then only the MTP tensors.

    python tools/mtp_fetch.py inventory --out DIR          # headers only (a few KB per shard)
    python tools/mtp_fetch.py fetch --out DIR [--only SUBSTR]  # the MTP tensors themselves, resumable
    python tools/mtp_fetch.py verify --out DIR             # the fetched tensors against SHA256 (exit 3: bad ones)

`fetch` writes one raw file per tensor plus `mtp-manifest.json` (dtype, shape, source shard, byte range,
sha256). It never downloads anything but the ranges named in the headers. Nothing here runs a model.

#327: a mirror or proxy that ignores the Range header answers 200 with the whole shard - its JSON header and
unrelated tensors - and a proxy may cut that to the requested length, so neither the size nor a hash of what was
downloaded catches it (the drafter then accepts nothing, silently).  A range read therefore needs a 206 whose
Content-Range is the range asked for, and every tensor of the pinned revision is checked against SHA256 below.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strata_control import tls  # noqa: E402

tls.install()       # #3722: the operating system's certificates

# #214: a fixed commit of the checkpoint (its `sha` from https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next
# on 2026-09-30), so every install reads the same tensors; STRATA_MTP_REVISION overrides it (e.g. main).  When the
# repository no longer has it, reading stops (resolve_repo) unless STRATA_ALLOW_UNPINNED=1.
PINNED_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
REVISION = os.environ.get("STRATA_MTP_REVISION") or PINNED_REVISION
# #495: HF_ENDPOINT (a mirror, e.g. https://hf-mirror.com) serves the same revision; the SHA256 checks below still apply
HF_ENDPOINT = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or "https://huggingface.co"
REPO = HF_ENDPOINT + "/Qwen/Qwen3.8-Flash-Next/resolve/%s/" % REVISION
PINNED = HF_ENDPOINT + "/Qwen/Qwen3.8-Flash-Next/resolve/%s/" % PINNED_REVISION   # SHA256's revision
# STRATA_SOURCE=modelscope (setup's --source): the same checkpoint from ModelScope.  It serves the repository's current
# files, not a pinned revision, so REPO and PINNED are one URL here: every tensor is still checked against the pinned
# revision's SHA256 below, so a tensor that differs from it is caught, not used.
if (os.environ.get("STRATA_SOURCE") or "").strip().lower() in ("ms", "modelscope"):
    MS_ENDPOINT = (os.environ.get("MODELSCOPE_ENDPOINT") or "").strip().rstrip("/") or "https://www.modelscope.cn"
    REPO = PINNED = MS_ENDPOINT + "/models/Qwen/Qwen3.8-Flash-Next/resolve/master/"
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8, "I32": 4}
BAD = 3                                             # `verify`'s exit code: a tensor is missing or corrupt

# #327: sha256 of every MTP tensor's bytes at PINNED_REVISION (two independent fetches agree, and 29 of them were read
# again from that revision; the two expert tensors' first and last 16 MiB), so a mirror that ignored the
# Range header (it sent the shard's start: its JSON header) is caught, and an install it corrupted is fetched again.
SHA256 = {
    "mtp.fc_embedding.weight":
        "241d6e87ded317a61b1aeb3ca4e28c53f1caba09ad5be66092e91fe5e3e14751",
    "mtp.fc_hidden.weight":
        "97c8e427ad0177e2c713ed24a397d815b523eda15ef453f1ae6463f6c70a8f96",
    "mtp.hyper_connection_mixer.hc_norm.weight":
        "ce948a398e15aeaa403c424cda34e6eed296ab2570d591bb5ff3da642b255d64",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight":
        "c633d2841fda463e82045e0b93fb8cb8d511ec0c9eda616020f01aca18d71947",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight":
        "6565285f0ba2923703252763cfc11d8a9bda40ab42e7548e12464b04dbdbec13",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight":
        "c8afff64e7f54412d02120097c4255033c55245e1c1eed84cc630d428ccb864c",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight":
        "1fb619aeaaf637ccd70f46f1f8f8116e6a46448cb43026b246de8ee8286ed809",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight":
        "e80ac8bf3e270e9d5a949ec7c913329dc8de017d69851334820fae7b8c86a7f5",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight":
        "f80f1fd44e1183e7764ea496b11ff89ead9dd758f861d4e79a770bf54956b1a5",
    "mtp.layers.0.mlp.experts.down_proj":
        "539c46234f73bac70450f731f40e715f77ac32faf77def0f3f800e58aaabbbc0",
    "mtp.layers.0.mlp.experts.gate_up_proj":
        "839e98f9642ebc884b178abd2090b6dc8b0bcd81ed92d7f7a08314c6a68f96d5",
    "mtp.layers.0.mlp.gate.weight":
        "f0d35fa41b360a08bc0c5b1e5766aa42df960291716ee189208b5d2599462c20",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight":
        "d2ced8c81367cc494599d1a8cdc279c3dd6579b628ded05acd7495e99cee01d8",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight":
        "f4a81109165853b3b64f36d809f98094b599763364070f5fa079200e1c386d4c",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight":
        "172586861eea53efbff10afc43ded5b6e49c83879e9efc01e089c652bbb0a156",
    "mtp.layers.0.mlp.shared_expert_gate.weight":
        "acdd306d9996d356e4667a16ff63af6fef086fc7f570f1bc00bbe882a2a93f6e",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight":
        "c5782dd92b479a1f8ef43d683fe9c917fd9e8fba81f23c1aae8817bfe4b476a6",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight":
        "21a504876b5e7b788301557c68dfb757999430eba36c66ec8ab96a63604dbfd1",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight":
        "b71240036dea954e034d5ae04e097bcd7ac51cb28a44e4c93d198af3447278a1",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight":
        "40756866bf2a5e82c28e407f717b6fa07614d593f78e79c7ceb2cd4ffe6dda94",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight":
        "2befafae6e944c1fc651fd73d4d527eb6f138a6c1679065b003128ee3150e859",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight":
        "2db621f6a08b4085756fc69d60a4a1bd1d7fbf5d406456d2477075842efaae54",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight":
        "afaf59f68e251939970ffdeacad0667dffc657a522bea62cdc44a52816fc49d6",
    "mtp.layers.0.self_attn.k_norm.weight":
        "8f924b0357cf936e14fc34518205d000e1423aa60626827df81b0804fae8cdfd",
    "mtp.layers.0.self_attn.k_proj.weight":
        "1a295dc2d6e633e808aeb19e00c97856a1817bbfcc83f5b610e2bfcacfcb8c38",
    "mtp.layers.0.self_attn.o_proj.weight":
        "9d2fcf0ac6ac19de87f638f710cfd1af6ad14141e75b7d3d5d29fbd06f374219",
    "mtp.layers.0.self_attn.q_norm.weight":
        "57957aa1e7c5a3ac628bff0ec5522ad55a39b8c0b8bd2d324ae640e3771617ef",
    "mtp.layers.0.self_attn.q_proj.weight":
        "5220aef3d288f40759f7356af9ef453e5ac0256a38749a9bb803a4e9db58150d",
    "mtp.layers.0.self_attn.v_proj.weight":
        "39cb12f5424e965fa71fb81db0ee230c701b7b1e0b72c36641fde7e0e41bff41",
    "mtp.pre_fc_norm_embedding.weight":
        "04c4a570850e06f2d8913da8220d54d4c7f87db6eb6d45480b938e8ba41d6a86",
    "mtp.pre_fc_norm_hidden.weight":
        "b933b44559879449b1baf2af8385c5c237347307e3fd452394870cb53e6eab6d",
}


# Hugging Face answers a /resolve/ address with a redirect to a signed CDN address (valid for about an hour).  Every range of a
# tensor used to go through that redirect and a fresh connection to it; the CDN address of the first answer is reused now
# (and forgotten after any error, so the next try asks again).
CDN_TTL = 600
_cdn = {}
_cdn_lock = threading.Lock()


def _final(url):
    with _cdn_lock:
        hit = _cdn.get(url)
    return hit[0] if hit and hit[1] > time.monotonic() else url


def get(url, start=None, end=None, retries=6):
    for attempt in range(retries):
        try:
            used = _final(url)
            req = urllib.request.Request(used, headers={"User-Agent": "strata-mtp-fetch"})
            if start is not None:
                req.add_header("Range", "bytes=%d-%d" % (start, end))
            with urllib.request.urlopen(req, timeout=120) as r:
                if start is not None:
                    check_range(r.status, r.headers.get("Content-Range"), start, end)
                data = r.read()
                landed = getattr(r, "geturl", lambda: url)()
            if start is not None and len(data) != end - start + 1:
                raise IOError("short range read: %d of %d" % (len(data), end - start + 1))
            if used == url and landed and landed != url:
                with _cdn_lock:
                    _cdn[url] = (landed, time.monotonic() + CDN_TTL)
            return data
        except Exception as e:  # network errors are retried, then surfaced
            with _cdn_lock:
                _cdn.pop(url, None)
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 16))
            print("retry %s: %s" % (url, e), file=sys.stderr)


class Progress:
    """One line a couple of seconds, `MTP draft layer: 1.23 / 5.04 GB (24%)`: the form of setup's own download line, so the app's
    progress parser (tools/strata_mcp.py install_progress) shows it.  `done` counts what an earlier run already has."""
    LABEL = "MTP draft layer"

    def __init__(self, total, done=0, every=2.0, out=None):
        self.total, self.done, self.every, self.out = total, done, every, out or sys.stderr
        self.last = 0.0
        self.lock = threading.Lock()

    def add(self, n):
        with self.lock:
            self.done += n
            self.show()

    def show(self, force=False):
        now = time.monotonic()
        if not force and now - self.last < self.every:
            return
        self.last = now
        if self.total > 0:
            done = max(0, min(self.done, self.total))
            print("  %s: %6.2f / %.2f GB (%d%%)" % (self.LABEL, done / 1e9, self.total / 1e9, 100 * done // self.total),
                  file=self.out, flush=True)


def lock_folder(out, wait=900.0):
    """Only one fetch writes a folder: a second one (a retry while the first still runs, #3257) waits for the first and then finds
    its tensors done.  Returns the open lock file (keep it open), or exits when the first does not end in `wait` seconds."""
    os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, ".mtp_fetch.lock"), "a+b")
    end = time.monotonic() + wait
    told = False
    while True:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            if time.monotonic() > end:
                sys.exit("another MTP download is still writing %s: wait for it or end it (Task Manager: python.exe), then try again" % out)
            if not told:
                print("another MTP download is writing %s: waiting for it to end" % out, file=sys.stderr, flush=True)
                told = True
            time.sleep(2)


def exit_with_parent():
    """STRATA_EXIT_WITH_PARENT=1 (set by setup.py for the tools it runs): end when the process that started this one ends, so a cancelled
    or killed setup does not leave a download running on its own (#3257: taskkill /T does not reach a grandchild whose parent is gone)."""
    if (os.environ.get("STRATA_EXIT_WITH_PARENT") or "").strip() != "1":
        return
    parent = os.getppid()
    if parent <= 1:
        return
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        h = k32.OpenProcess(0x00100000, False, parent)       # SYNCHRONIZE: a handle keeps the answer true if the id is reused
        if not h:
            return                                           # no access (or already gone): nothing to watch
        gone = lambda: k32.WaitForSingleObject(h, 0) == 0
    else:
        gone = lambda: os.getppid() != parent

    def watch():
        while not gone():
            time.sleep(1.5)
        print("the process that started this download has ended: stopping", file=sys.stderr, flush=True)
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


def connections():
    """Ranges read at once: STRATA_DL_CONNECTIONS (the setting of setup's downloads), else 16; 1 reads them one by one."""
    try:
        n = int((os.environ.get("STRATA_DL_CONNECTIONS") or "").strip() or 16)
    except ValueError:
        n = 16
    return max(1, min(n, 32))


def ranged(url, start, last, chunk, workers):
    """The bytes start..last of `url` in order, read in `chunk`-sized ranges with `workers` of them in flight.  Each range is
    checked by get() (a 206 with the range asked for), so a server that ignores Range still fails here."""
    spans = []
    pos = start
    while pos <= last:
        spans.append((pos, min(pos + chunk - 1, last)))
        pos = spans[-1][1] + 1
    if workers <= 1 or len(spans) <= 1:
        for a, b in spans:
            yield get(url, a, b)
        return
    todo = iter(spans)
    window = collections.deque()
    # the ranges are handed on in order, so one slow range holds the oldest slot; the connections that are done go on with
    # the next ranges meanwhile (up to `depth` ahead, at most ~512 MiB in memory) instead of waiting for it
    depth = max(workers, min(3 * workers, (512 << 20) // max(chunk, 1)))
    with concurrent.futures.ThreadPoolExecutor(min(workers, len(spans))) as ex:
        for a, b in todo:
            window.append(ex.submit(get, url, a, b))
            if len(window) >= depth:
                break
        while window:
            data = window.popleft().result()
            for a, b in todo:
                window.append(ex.submit(get, url, a, b))
                break
            yield data


def check_range(status, content_range, start, end):
    """#327: a range read is only what was asked for when the server says so: 206 and `bytes START-END/...`."""
    if status != 206:
        raise IOError("range request not honoured: HTTP %s instead of 206 (a mirror or proxy that ignores the Range "
                      "header sends the whole file)" % status)
    want = "bytes %d-%d/" % (start, end)
    if not (content_range or "").strip().startswith(want):
        raise IOError("range request answered with Content-Range %r, not %s..." % (content_range, want))


def pinned():
    """The tensors' hashes are the pinned revision's: they apply when this run reads that revision."""
    return REPO == PINNED


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def resolve_repo():
    """REPO.  When the pinned revision is gone from the repository (a 404 on its index) it stops, unless
    STRATA_ALLOW_UNPINNED=1: then the current files are read (their tensors are still checked against the pinned SHA256)."""
    global REPO
    try:
        req = urllib.request.Request(REPO + "model.safetensors.index.json", method="HEAD",
                                     headers={"User-Agent": "strata-mtp-fetch"})
        urllib.request.urlopen(req, timeout=120).close()
    except urllib.error.HTTPError as e:
        if e.code == 404 and "/resolve/%s/" % REVISION in REPO:
            if (os.environ.get("STRATA_ALLOW_UNPINNED") or "").strip() != "1":
                sys.exit("the pinned version of the checkpoint (%s) is no longer on Hugging Face: update Strata" % REVISION)
            print("STRATA_ALLOW_UNPINNED=1: the checkpoint's pinned revision %s is gone: reading its current files "
                  "(main); every tensor is still checked against the pinned SHA256" % REVISION, file=sys.stderr)
            REPO = REPO.replace("/resolve/%s/" % REVISION, "/resolve/main/")
    except OSError:
        pass                                        # no answer: get() retries and reports it
    return REPO


def shard_header(shard):
    url = REPO + shard
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    header = json.loads(get(url, 8, 8 + n - 1))
    return 8 + n, header


def inventory(out):
    index = json.loads(get(REPO + "model.safetensors.index.json"))["weight_map"]
    mtp = {k: v for k, v in index.items() if k.startswith("mtp.")}
    rows, total = [], 0
    shards = sorted(set(mtp.values()))
    # one small request pair per shard (28 of them): side by side, not one after the other (28 s -> 4 s)
    with concurrent.futures.ThreadPoolExecutor(max(1, min(connections(), 8, len(shards)))) as ex:
        headers = list(ex.map(shard_header, shards))
    for shard, (base, header) in zip(shards, headers):
        for name, meta in header.items():
            if name in mtp and mtp[name] == shard:
                a, b = meta["data_offsets"]
                rows.append(dict(name=name, shard=shard, dtype=meta["dtype"], shape=meta["shape"],
                                 start=base + a, end=base + b - 1, bytes=b - a))
                total += b - a
    missing = sorted(set(mtp) - set(r["name"] for r in rows))
    if missing:
        sys.exit("tensors named in the index but absent from their shard headers: %s" % missing)
    rows.sort(key=lambda r: r["name"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "mtp-inventory.json"), "w", encoding="utf-8") as f:
        json.dump(dict(repo=REPO, total_bytes=total, tensors=rows), f, indent=1)
    lines = ["# MTP block in the BF16 checkpoint", "", "%d tensors, %.3f GB, in %d shards." % (len(rows), total / 1e9, len(set(r["shard"] for r in rows))), "",
             "| tensor | dtype | shape | MB |", "|---|---|---|---:|"]
    for r in rows:
        lines.append("| `%s` | %s | %s | %.1f |" % (r["name"], r["dtype"], "x".join(map(str, r["shape"])), r["bytes"] / 1e6))
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "mtp-inventory.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return rows


def fetch(out, only):
    inv_path = os.path.join(out, "mtp-inventory.json")
    tdir = os.path.join(out, "tensors")
    inv = None
    if os.path.exists(inv_path):
        try:
            with open(inv_path, encoding="utf-8") as f:
                inv = json.load(f)
        except ValueError:
            inv = None
    # The byte ranges belong to one revision: an inventory read from another repository or revision is read again,
    # and a tensor fetched from there is not resumed - it is kept only when it is the pinned revision's (sha256).
    same = isinstance(inv, dict) and inv.get("repo") == REPO
    rows = inv["tensors"] if same else inventory(out)
    os.makedirs(tdir, exist_ok=True)
    manifest = []
    workers = connections()
    chunk = (16 if workers > 1 else 64) << 20
    todo = [r for r in rows if not only or only in r["name"]]
    have_all = 0
    for r in todo:
        tp = os.path.join(tdir, r["name"] + ".bin")
        if os.path.exists(tp):
            have_all += min(os.path.getsize(tp), r["bytes"])
    progress = Progress(sum(r["bytes"] for r in todo), have_all)
    progress.show(force=True)
    for r in todo:
        path = os.path.join(tdir, r["name"] + ".bin")
        want = SHA256.get(r["name"]) if pinned() else None
        if os.path.exists(path) and not same and (
                want is None or os.path.getsize(path) != r["bytes"] or sha256_of(path) != want):
            os.remove(path)
        attempt, net_fails = 0, 0
        while attempt < 2:
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have > r["bytes"]:
                os.remove(path)
                have = 0
            try:
                with open(path, "ab") as f:
                    pos = r["start"] + have
                    for data in ranged(REPO + r["shard"], pos, r["end"], chunk, workers):
                        f.write(data)
                        pos += len(data)
                        progress.add(len(data))
            except (OSError, urllib.error.URLError) as e:
                # #3214: a timeout or a dropped connection mid-tensor ends only this try: what was written stays, the next try
                # continues after it with fewer connections (a link that drops 16 at once often holds 4)
                net_fails += 1
                if net_fails > 3:
                    sys.exit("%s: the download keeps failing (%s). Check the connection and try again - what is downloaded is kept; "
                             "if Hugging Face is slow to reach, `--source modelscope` reads the same files elsewhere" % (r["name"], e))
                workers = max(4, workers // 2)
                print("%s: %s - continuing with %d connections (try %d)" % (r["name"], e, workers, net_fails + 1), file=sys.stderr, flush=True)
                time.sleep(min(5 * net_fails, 20))
                continue
            if os.path.getsize(path) != r["bytes"]:
                sys.exit("%s: size %d != %d" % (path, os.path.getsize(path), r["bytes"]))
            digest = sha256_of(path)
            if want is None or digest == want:
                break
            progress.done -= os.path.getsize(path)
            os.remove(path)                         # wrong bytes (#327): the whole tensor again, once
            attempt += 1
            if attempt > 1:
                sys.exit("%s: sha256 %s is not the checkpoint's %s - the download source sends wrong data (a mirror "
                         "that ignores Range requests?)" % (r["name"], digest, want))
            print("%s: wrong bytes (sha256 %s...), fetching it again" % (r["name"], digest[:12]), file=sys.stderr)
        manifest.append(dict(r, file=os.path.relpath(path, out), sha256=digest))
        print("%s done" % r["name"], file=sys.stderr, flush=True)
    progress.show(force=True)
    with open(os.path.join(out, "mtp-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


def verify(out):
    """#327: the fetched tensors against SHA256 -> the names that are missing or wrong.  [] when they are right, or
    when there is nothing to check (no tensors/, or STRATA_MTP_REVISION names another revision).  A tensor's verdict
    is kept in tensors/verified.json by size and mtime, so a later run hashes only what changed."""
    tdir = os.path.join(out, "tensors")
    if not pinned() or not os.path.isdir(tdir):
        return []
    stamp_path = os.path.join(tdir, "verified.json")
    try:
        with open(stamp_path, encoding="utf-8") as f:
            stamps = json.load(f)
    except (OSError, ValueError):
        stamps = {}
    bad, good = [], {}
    for name, want in sorted(SHA256.items()):
        path = os.path.join(tdir, name + ".bin")
        if not os.path.exists(path):
            bad.append(name)
            continue
        st = os.stat(path)
        key = [st.st_size, st.st_mtime_ns, want]
        if stamps.get(name) != key and sha256_of(path) != want:
            bad.append(name)
            continue
        good[name] = key
    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            json.dump(good, f, indent=1)
    except OSError:
        pass
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inventory", "fetch", "verify"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    a = ap.parse_args()
    if a.cmd == "verify":                           # offline: the hashes are the pinned revision's
        bad = verify(a.out)
        for name in bad:
            print("MTP tensor missing or corrupt: %s" % name, file=sys.stderr)
        sys.exit(BAD if bad else 0)
    exit_with_parent()
    lock = lock_folder(a.out) if a.cmd == "fetch" else None
    resolve_repo()
    inventory(a.out) if a.cmd == "inventory" else fetch(a.out, a.only)
    if lock:
        lock.close()


if __name__ == "__main__":
    main()
