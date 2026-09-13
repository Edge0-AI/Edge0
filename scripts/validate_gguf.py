"""Local acceptance probe. Records actual tokens, timing and physical footprint.

No reference downloads. Use --inspect first; generation runs only when requested.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import threading
import time


def physical_footprint(pid=None, peak=False):
    # Darwin rusage_info_v4: current footprint is uint64 field 7, lifetime
    # maximum is field 28. This includes compressed and swapped anonymous RAM.
    class Usage(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 35)]
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    info = Usage()
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    if lib.proc_pid_rusage(pid or os.getpid(), 4, ctypes.byref(info)):
        raise OSError("proc_pid_rusage failed")
    return info.values[28 if peak else 7]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--memory-pid", type=int, help="inspect a running validation process only")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--prompt", default="Explain why the sky is blue.")
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--max-new", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", default="/tmp/edge0-gguf-validation.json")
    parser.add_argument("--cache-mib", type=int, help="override the architecture's default packed cache size")
    parser.add_argument('--execution', choices=('packed', 'decoded-baseline'), default='packed')
    args = parser.parse_args()
    if args.memory_pid:
        print(json.dumps({"physical_footprint_bytes": physical_footprint(args.memory_pid),
                          "peak_physical_footprint_bytes": physical_footprint(args.memory_pid, peak=True)}))
        return
    if args.model_path is None:
        parser.error("model_path is required unless --memory-pid is used")
    os.environ["MLX_ENABLE_TF32"] = "0"
    if args.execution == 'decoded-baseline':
        import edge0.streaming.gguf as streaming
        from gguf_decoded_baseline import Weights, Experts
        streaming.GGUFWeights, streaming.GGUFExperts = Weights, Experts
    from edge0 import AutoEngine
    import mlx.core as mx
    import numpy as np
    report = {"model_path": str(Path(args.model_path).expanduser()), "runs": [], "execution": args.execution}
    def save_report():
        # Preserve progress even if the process is later killed for memory pressure.
        destination = Path(args.output)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    stop = threading.Event()
    def monitor():
        peak = 0
        while not stop.wait(0.1):
            peak = max(peak, physical_footprint(peak=True))
            report["peak_physical_footprint_bytes"] = peak
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    engine = None
    try:
        started = time.perf_counter()
        cache_options = {} if args.cache_mib is None else {"weight_cache_bytes": args.cache_mib * 1024**2}
        engine = AutoEngine.from_pretrained(model_path=args.model_path, tokenizer_path=args.tokenizer_path,
            **cache_options)
        report.update(build_s=time.perf_counter() - started, tensors=len(engine.source.tensors),
            architecture=engine.cfg.architecture, experts=engine.cfg.moe_spec.num_experts,
            top_k=engine.cfg.moe_spec.top_k, bytes_read_after_build=engine.source.bytes_read,
            device=mx.device_info())
        save_report()
        if args.inspect:
            return
        ids = engine.encode_chat([{"role": "user", "content": args.prompt}])
        if args.prompt_tokens:
            if args.prompt_tokens < 1:
                raise ValueError("prompt-tokens must be positive")
            padding = max(0, args.prompt_tokens - len(ids) - 8)
            for _ in range(8):
                content = "Background terms: " + "sky " * padding + "\n\n" + args.prompt
                ids = engine.encode_chat([{"role": "user", "content": content}])
                if len(ids) == args.prompt_tokens:
                    break
                padding += args.prompt_tokens - len(ids)
                if padding < 0:
                    raise ValueError("requested token length is shorter than the chat prompt")
            if len(ids) != args.prompt_tokens:
                raise ValueError("could not construct exact-length tokenized chat prompt")
        report["prompt_ids"] = ids
        report["rendered_prompt"] = engine._tok.decode(ids)
        save_report()
        for repeat in range(args.repeat):
            engine.reset()
            started = time.perf_counter()
            def progress(done, total):
                report["in_progress"] = dict(run=repeat, phase="prefill", tokens=done,
                    total=total, elapsed_s=time.perf_counter() - started)
                save_report()
                print(f"prefill {done}/{total}", flush=True)
            engine.prefill(ids, on_progress=progress)
            prefill_s = time.perf_counter() - started
            logits = np.array(engine.next_logits())
            np.save(str(args.output) + f".run{repeat}.logits.npy", logits)
            if not np.isfinite(logits).all():
                raise ValueError("non-finite prefill logits")
            output = []
            started = time.perf_counter()
            for _ in range(args.max_new):
                token = int(np.argmax(logits))
                if engine.cfg.gen.is_eos(token):
                    break
                output.append(token)
                print(engine._tok.decode([token]), end="", flush=True)
                logits = np.array(engine.step(token))
                report["in_progress"] = dict(run=repeat, phase="decode",
                    generated_ids=list(output), elapsed_s=time.perf_counter() - started)
                save_report()
            elapsed = time.perf_counter() - started
            report["runs"].append(dict(prefill_s=prefill_s, decode_s=elapsed,
                decode_tokens_per_s=len(output) / elapsed if elapsed else 0,
                generated_ids=output, generated_text=engine._tok.decode(output), stats=engine.stats(),
                physical_footprint_bytes=physical_footprint(), mlx_peak_bytes=mx.get_peak_memory()))
            report["peak_physical_footprint_bytes"] = max(report.get("peak_physical_footprint_bytes", 0), physical_footprint())
            report.pop("in_progress", None)
            save_report()
            print(json.dumps(report["runs"][-1]), flush=True)
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        if engine is not None:
            report["final_stats"] = engine.stats()
            engine.close()
        report["peak_physical_footprint_bytes"] = physical_footprint(peak=True)
        stop.set()
        watcher.join()
        save_report()
        print(f"\nReport: {args.output}", flush=True)


if __name__ == "__main__":
    main()
