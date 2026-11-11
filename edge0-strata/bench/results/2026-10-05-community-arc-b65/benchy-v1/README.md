# Public benchy-v1 prompt fixtures, Arc Pro B65 Gen4

Measured 2026-10-06 using the isolated patched v0.1.40 IQ2_XS/8K/INT8KV/MTP4
profile, with the retained model/driver/512 batch/3072 MiB reserve/four workers.
Exact 20/2185 raw input IDs are from maxfridbe/Strata_B70 at
79ad9d5aff716292602895b19860c7661bacc5ed, sycl/bench/v1. Our qualified native
serving protocol is used with those fixtures; this is not an unmodified
sycl/benchy.sh run. See [method and reproduction](METHOD.md) and
[build/model/hardware details](../README.md).

Three fresh native engine processes each run the long fixture first, then the
short fixture, both greedy with 256 outputs, followed by an excluded exact
canary. Startup is excluded; the first long decode includes native graph capture.
OS page cache remains warm/retained, prompt reuse/adaptation off. STRATA_TRACE=1
provides numeric path lines; no GPU profiler or device-event instrumentation.
All six measured outputs match their qualified reference token hashes,
and all repetitions match each other. All three native shutdowns exit0.

Each rate/time cell is median [minimum–maximum] of three repetitions; rates
tok/s and times seconds. Separate native prompt/decode timers are used.

|Input|Generated|Runs|Reused|Prompt tok/s|Decode tok/s|TTFT s|Total s|Draft acceptance|
|---:|---:|---:|---:|---|---|---|---|---:|
|20|256|3|0|52.92 [52.38–54.26]|47.71 [47.09–49.39]|0.41 [0.40–0.42]|5.74 [5.55–5.82]|72.6%|
|2185|256|3|0|308.81 [305.20–318.05]|47.58 [47.22–48.26]|7.13 [6.93–7.22]|12.46 [12.18–12.58]|77.4%|

[All records](results.json), [CSV](results.csv), [summary](summary.json),
[native timing lines](native-timings.log), [lifecycle](lifecycle.json),
[health](health-summary.json). All nine requests, including three excluded
canaries, are retained numerically. No generated text is published.

These prompts and output lengths differ from the five-task/640-output
benchmark, so their rate is not an equivalent-workload decode improvement.
