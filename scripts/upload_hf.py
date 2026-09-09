"""Upload edge0 model directories to Hugging Face via hf-mirror (no xet).

Two mirror quirks must be worked around:
  1. hf-mirror 308-redirects API calls to huggingface.co and the default
     client drops the Authorization header (and body) on cross-host
     redirects -> 401. Patched by re-attaching them.
  2. The mirror rewrites callback hosts in LFS batch responses to
     "hf-mirror.org" (a nonexistent domain). Those callbacks (complete_multipart,
     verify) are re-rewritten to huggingface.co, which is reachable through
     the local proxy (tiny POSTs, speed irrelevant). Bulk S3 part uploads go
     direct (NO_PROXY) at full speed.
  3. The xet upload path fails (xet-write-token goes straight to
     huggingface.co) -> disabled.

Usage:
    export HF_TOKEN=...
    export EDGE0_8B_MODEL=/path/to/edge0-8b-checkpoint
    python scripts/upload_hf.py [edge0-8b|edge0-35b]
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault(
    "NO_PROXY",
    "hf-mirror.com,cdn-lfs.huggingface.co,cdn-lfs-us-1.huggingface.co,"
    "s3-accelerate.amazonaws.com,hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com",
)
os.environ["no_proxy"] = os.environ["NO_PROXY"]

import httpx  # noqa: E402

_orig_handle = httpx.HTTPTransport.handle_request


def _handle(self, request):
    resp = _orig_handle(self, request)
    hops = 0
    while resp.has_redirect_location and hops < 5:
        hops += 1
        loc = resp.headers.get("location", "")
        url = httpx.URL(loc if loc.startswith("http") else str(request.url.join(loc)))
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        method = "GET" if resp.status_code in (301, 302, 303) else request.method
        body = request.content if method != "GET" else None
        resp = _orig_handle(self, httpx.Request(method, url, headers=headers, content=body))
    return resp


httpx.HTTPTransport.handle_request = _handle

from huggingface_hub import lfs as _lfs  # noqa: E402
from huggingface_hub.hf_api import HfApi  # noqa: E402

# mirror bug: callback hosts rewritten to hf-mirror.org -> point them back at
# huggingface.co (proxied, tiny requests)
_orig_fix = _lfs.fix_hf_endpoint_in_url


def _fix(url, endpoint):
    url = url.replace("https://hf-mirror.org", "https://huggingface.co")
    return _orig_fix(url, endpoint)


_lfs.fix_hf_endpoint_in_url = _fix

HfApi._validate_yaml = lambda self, content, repo_type, token=None: None

REPOS = {
    "edge0-8b": ("EDGE0_8B_MODEL", "Edge0/Edge0-8B-A1B-preview"),
    "edge0-35b": ("EDGE0_35B_MODEL", "Edge0/Edge0-35b-a3b-preview"),
}


def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else "edge0-8b"
    env_name, repo_id = REPOS[tier]
    local_dir = os.environ.get(env_name)
    if not local_dir:
        raise SystemExit(
            f"set {env_name} to the local checkpoint directory before "
            "uploading")
    local_dir = os.path.abspath(os.path.expanduser(local_dir))
    if not os.path.isdir(local_dir):
        raise SystemExit(f"checkpoint directory not found: {local_dir}")
    api = HfApi()
    print(f"uploading {local_dir} -> {repo_id} via {os.environ['HF_ENDPOINT']} (no xet)", flush=True)
    names = sorted(
        f for f in os.listdir(local_dir)
        if not f.startswith(".") and ".bak_vision" not in f and f != "README.md"
    )
    # README usually already uploaded; skip leftovers handled below
    files = [n for n in names if os.path.isfile(os.path.join(local_dir, n))]
    for i, name in enumerate(files):
        path = os.path.join(local_dir, name)
        print(f"[{i+1}/{len(files)}] {name} ({os.path.getsize(path)} bytes)", flush=True)
        r = api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  -> {r}", flush=True)
    print("done:", repo_id)


if __name__ == "__main__":
    main()
