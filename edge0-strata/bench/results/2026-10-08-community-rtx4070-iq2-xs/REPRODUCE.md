# Reproduce the RTX 4070 12 GB / 32 GB DDR4 IQ2_XS result

**Scope:** Windows 11, NVIDIA RTX 4070 12 GB, 32 GB DDR4-3200, i5-12600KF.
Historical measurement: 2026-10-08; five independent CLI processes; 28,912 prompt
tokens, 1,024 generated tokens; median **48.23 accepted decode tok/s** (E4/S24).

This is a **reproduction package**, not a claim that any 4070 gets 48.23 tok/s.
Workload content, disk/PCIe characteristics, runtime cache state, compiler
toolchain, and the 20 GiB resident-memory policy can affect performance.
The reference engine is a modified Adrian-derived build, **not stock Strata**.

## 1. Exact source code (published, full source tree)

Source: [Hcl192088/Strata, pinned IQ2 benchmark engine snapshot](https://github.com/Hcl192088/Strata/tree/benchmark-iq2-source-9ec3806)

Immutable source commit:

```text
9ec3806058cf32ab27a55e4377daf7cf0d087dec
```

A full GitHub archive is also available from
`https://github.com/Hcl192088/Strata/archive/9ec3806058cf32ab27a55e4377daf7cf0d087dec.zip`.
The commit is from the [AdrianBM96/Strata3060](https://github.com/AdrianBM96/Strata3060)
line. The source branch above is a **verbatim tree pinned to that historical
commit**, not a reconstruction based on latest upstream and not a cherry-pick.
It is deliberately kept separate from this benchmark-data PR, to avoid
improperly presenting Adrian's engine modifications as new upstream patches.

On Windows PowerShell:

```powershell
git clone --single-branch --branch benchmark-iq2-source-9ec3806 https://github.com/Hcl192088/Strata.git .\strata-iq2-source
cd .\strata-iq2-source
git rev-parse HEAD
# Must print 9ec3806058cf32ab27a55e4377daf7cf0d087dec.
```

The source's `setup.py` pins a llama.cpp dependency commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, Hugging Face model revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`, and Python dependency
versions through `requirements.txt`.

## 2. Build and prepare the model

On a similar Windows/NVIDIA PC with sufficient free SSD storage, Git,
Python 3.10+, Windows Visual Studio Build Tools, and NVIDIA CUDA Toolkit,
the historical source has its own installer/build and pack-generation pipeline:

```powershell
py -3 setup.py --family qwen --model IQ2_XS --context 100000 --low-ram resident --build --no-start --yes
```

This invokes the pinned source's own `build_engine` and model setup logic. For
RTX 4070 the target compute capability is `sm_89`. The build uses CMake/Ninja,
CUDA and llama.cpp/ggml. The official full-model GGUF is fetched by the source
setup from the pinned ISTA-DASLab repository revision. Its `iq_pack.py` can
generate `packs/iq2_xs` including `experts.bin`; the source's
`mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py` prepare `mtp/rt`.

**Important:** This setup produces a *candidate*, not an attested byte-for-byte
replica of the measurement. Verify every SHA-256 in `provenance.json`; on
mismatch, inspect toolchain/pack provenance rather than claiming an exact run.
The historical binary's Windows compiler, CUDA toolkit and build flags were
not captured sufficiently to promise identical executable bytes.

## 3. Prepare a *local* expert profile and prompt

The original benchmark used `--expert-profile profiles/learned-heart.bin`.
**This is a Strata expert-cache profile, not an external trained model that
every tester needs to download.** The pinned source includes two ways to
create your own profile:

1. **Built-in learning (preferred for a representative workload):** start
   `strata --serve` with an existing `--expert-profile`, the adaptive tier
   enabled (`--adapt-every 4 --adapt-swaps 24`), and
   `--expert-profile-save <path>`. Interact with the server using your
   usual workload; it writes its learned ranking periodically (default ten
   minutes) or when you enter `QUIT` in its stdin command interface.
   The saved file can be loaded via `--expert-profile` during benchmarking.
   This is an **opt-in feature, not an automatic byproduct** of simply
   enabling expert caching.

2. **One-shot routing trace:** run the source engine with
   `--dump-routing <trace.bin>` on your own representative prompt, then:
   ```powershell
   py -3 .\tools\make_profile.py "D:\my-trace.bin" --reorder --out "D:\my-learned.bin"
   ```
   This alternative uses the bundled `data/expert-profile.bin` as the
   base ranking, moving frequently routed experts to the front.

You may also start with the repository's unlearned
`data/expert-profile.bin` to establish a baseline, without any learning.
The original **SHA-256 of `learned-heart.bin` remains in provenance.json for
historical identification**, but publishing the original file is *not*
necessary for an independent performance experiment.

**Do not publish the private original prompt.** Instead use your own text
(e.g. a locally available article or synthetic workload). The included
[`make_local_prompt.py`](make_local_prompt.py) generates a local
`neuro.tokens`-compatible token-ID file of **28,912 input tokens**, with no
network call and without publishing its contents:

```powershell
$Src = "D:\strata-iq2-source"
$Data = "D:\strata-iq2-data"
$Shard1 = "$Data\models\IQ2_XS\Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
$Profile = "$Src\data\expert-profile.bin"
$Prompt = "D:\strata-iq2-runs\my-prompt.tokens"

py -3 .\make_local_prompt.py --source-root $Src --gguf $Shard1 --text "D:\my-document.txt" --out $Prompt
```

To test a learned profile, replace `$Profile` with your own saved profile
file. **The chosen profile/training workload and prompt affect routing, MTP
acceptance and decode speed**, so results with different inputs must not be
presented as exact replays of the 2026-10-08 five-run sequence.

The report's `provenance.json` still provides full SHA-256 and file sizes for
model, pack, tokenizer and MTP assets. The historical binary's exact toolchain
was not captured. Recompilation may therefore produce a different binary hash.

## 4. Verify and run five local measurements

The companion [reproduce.py](reproduce.py) is a Python-standard-library
validator/harness. It accepts `--profile` and `--prompt-file` overrides so
no one needs the original private prompt or `learned-heart.bin` bytes.
Use the exact source commit, model and core launch settings, and your own
length-matched workload.

```powershell
$Exe = "$Src\engine\strata.exe"

# Verify matching model/MTP/pack files and the exact source checkout.
# --allow-binary-mismatch explicitly permits a different binary from recompiling
# the same source. It will be flagged in the output and is NOT byte-identical.
py -3 .\reproduce.py check --data-root $Data --engine $Exe --source-root $Src `
  --profile $Profile --prompt-file $Prompt --allow-binary-mismatch

# Keep cache/warm-up policy consistent across candidate/control experiments.
# This is a five-run comparable workload, NOT the original prompt replay.
py -3 .\reproduce.py run --data-root $Data --engine $Exe --source-root $Src `
  --profile $Profile --prompt-file $Prompt --allow-binary-mismatch `
  --runs 5 --out "D:\strata-iq2-runs\repeat01"
```

The full-hash `check` should be a separate preflight; the `run` command
uses size checks to avoid warming all large files by hashing them immediately
before a timed run. The runner captures the actual effective commands,
the prompt/profile SHA-256, original source commit, observed binary hash,
per-run stdout/stderr/exit, before/after GPU snapshots and the mean/median/range
of accepted decode throughput. Run-boundary snapshots are **not peak VRAM**.
Files remain local and logs may contain private token IDs.

**Historical data remain unchanged.** The public report accurately states
that the historical main run used its own profile and medical-topic prompt
(28,912 input tokens). Reproducing the *method* requires neither exact file.
The historical 48.23 tok/s median should be treated as a reference, not an
exact equality target for a different workload.

## 5. Comparison rule

Compare *accepted decode tok/s*, using the `decode ... tok/s` line emitted
by this exact engine; do not combine prompt processing time or speculative
proposals into the denominator. Use the per-run data and report the number of
valid samples, median and range. Any early termination, nonzero exit or output
token count different from 1,024 is reported as a failed run, not silently
excluded as a fast result.

Reference E4/S24 vector: `48.23, 47.98, 49.44, 48.83, 46.92` tok/s;
median `48.23` tok/s. The frozen E4/S32 reference median is `47.76` tok/s.
The E4/S24 improvement (`+0.98%`) **did not pass** the original `+1%`
promotion criterion.

**Reproduction terminology:** A community member with the same hardware
can rebuild the engine, create their own profile and private length-matched
prompt, then repeat the five-run **accepted decode tok/s methodology** without
access to any private user data. Different prompt/profile inputs are a
**comparable independent test**, not a byte-for-byte replay of the original
48.23 tok/s result. Report both the measured throughput and the differences.
