"""edge0 command-line entry point (``edge0``).

Commands:

    edge0 demo        one-command quickstart: find a checkpoint, generate
    edge0 serve       start the HTTP server (one model, queued generations)
    edge0 chat        one-shot prompt -> answer on the terminal
    edge0 models      list registered tiers and their default profiles
    edge0 convert-adapters   one-shot legacy npz -> safetensors migration
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from edge0 import models  # noqa: F401  (populates MODEL_REGISTRY)
from edge0.registry import MODEL_REGISTRY

# Tier name -> environment variable that locates that tier's checkpoint
# (no built-in paths: every machine resolves its own checkpoints).
TIER_ENV = {
    "edge0-35b": "EDGE0_35B_MODEL",
    "edge0-8b": "EDGE0_8B_MODEL",
}

DEMO_PROMPTS = {
    "edge0-35b": "Hello! Write one short sentence about the seaside.",
    "edge0-8b": "你好，用一句话介绍海滨城市。",
}


def cmd_models(args) -> int:
    for name in sorted(MODEL_REGISTRY):
        mod = MODEL_REGISTRY[name]
        cfg = mod.Config.from_pretrained(None)  # tier defaults
        print(f"{name}  (port {cfg.port}, target {cfg.target_tok_s} tok/s, "
              f"peak ≈ {cfg.peak_active_mem_mb:.0f} MB)")
        print(f"  experts={cfg.moe_spec.num_experts} "
              f"top_k={cfg.moe_spec.top_k} "
              f"quant={cfg.moe_spec.quant.bits}bit/"
              f"g{cfg.moe_spec.quant.group_size} "
              f"staged_n={cfg.options.staged_n} "
              f"prefill_full={cfg.options.prefill_full_layers} "
              f"hot={cfg.options.hot_per_layer}")
        pr = cfg.prerouter
        if pr is not None:
            owners = pr.owners
            owners_txt = (f"owners={owners[0]}..{owners[-1]} ({len(owners)})"
                          if owners else "owners=default")
            print(f"  prerouter: start={pr.start_layer} hidden={pr.hidden} "
                  f"dtype={pr.dtype} K={cfg.prerouter_top_k} {owners_txt} "
                  f"weights={pr.weights_file}")
        if cfg.lora:
            print(f"  lora: {cfg.lora} (r={cfg.lora_r} alpha={cfg.lora_alpha})")
    return 0


def _engine_kwargs(args) -> dict:
    """Map CLI flags to from_pretrained overrides (absent key = keep the
    tier default; explicit None/' ' disables)."""
    kw: dict = {}
    cache_mib = getattr(args, "cache_mib", None)
    if cache_mib is not None:
        from pathlib import Path
        path = getattr(args, "model_path", None)
        if not path or not Path(path).expanduser().is_file():
            raise ValueError("--cache-mib requires --model-path pointing to a GGUF checkpoint file")
        if cache_mib < 8:
            raise ValueError("--cache-mib must be at least 8 MiB (one packed chunk)")
        kw["weight_cache_bytes"] = cache_mib * 1024**2
    if getattr(args, "model_path", None):
        kw["model_path"] = args.model_path
        kw["tokenizer_path"] = args.tokenizer_path
    if getattr(args, "no_prerouter", False):
        kw["prerouter"] = None
    if getattr(args, "no_lora", False):
        kw["lora"] = ""
    return kw


def _resolve_model(args) -> tuple[str | None, str | None]:
    """Resolve the positional ``model`` argument.

    ``edge0 serve <model>`` accepts either a registered tier name
    (``edge0-35b`` / ``edge0-8b`` — checkpoint located via the matching
    ``EDGE0_<TIER>_MODEL`` environment variable) or a checkpoint path
    (tier auto-detected from the checkpoint's ``config.json``).

    Returns ``(model_dir, name)``; either may stay None to keep the
    ``--model-dir`` / ``--name`` behaviour.
    """
    model = getattr(args, "model", None)
    if getattr(args, "model_path", None):
        if model or args.model_dir or args.name:
            raise ValueError("--model-path cannot be combined with a positional model, --model-dir, or --name")
        return None, None
    if getattr(args, "tokenizer_path", None):
        raise ValueError("--tokenizer-path requires --model-path")
    if model and (args.model_dir or args.name):
        raise ValueError("positional model cannot be combined with --model-dir or --name")
    if not model:
        return args.model_dir, args.name
    if model in MODEL_REGISTRY:
        env = os.environ.get(TIER_ENV.get(model, ""))
        return env, model
    return model, None  # a path: tier auto-detected by the registry


def _missing_model_help(name) -> str:
    tier_env = TIER_ENV.get(name, "EDGE0_<TIER>_MODEL")
    return (
        f"[edge0] no checkpoint for {name or 'model'}.\n"
        f"Set {tier_env} to the checkpoint directory, e.g.\n"
        f"    export {tier_env}=/path/to/model\n"
        "or pass the checkpoint explicitly:\n"
        "    edge0 demo /path/to/model\n"
        "or point at any compatible checkpoint (tier auto-detected from"
        " config.json)."
    )


def _strip_thinking(text: str) -> str:
    """Strip a leading `` thinking... response`` reasoning block for display.

    The qwen chat template defaults to thinking mode, so a raw generation
    echoes the reasoning chain before the ``response`` marker.  The demo
    and chat commands show only the final answer unless ``--show-thinking``
    is given; the engine/server output is never modified.
    """
    m = re.search(r"\n\s*response\b", text)
    if m:
        return text[m.end():].strip()
    return text.strip()


def _display_text(text: str, show_thinking: bool) -> str:
    return text if show_thinking else _strip_thinking(text)


def cmd_demo(args) -> int:
    from edge0 import AutoEngine
    from edge0.server.chat import ChatMessage, ChatRequest, ChatSession

    model_dir, name = _resolve_model(args)
    if not getattr(args, "model_path", None) and (not model_dir or not os.path.isdir(model_dir)):
        print(_missing_model_help(name), file=sys.stderr)
        return 2
    engine = AutoEngine.from_pretrained(model_dir, name=name,
                                        **_engine_kwargs(args))
    try:
        tok = engine._tok
        if tok is None:
            raise SystemExit("model has no tokenizer; cannot demo")
        prompt = args.prompt or DEMO_PROMPTS.get(engine.name, "Hello!")
        req = ChatRequest(model=engine.name, messages=[
            ChatMessage(role="user", content=prompt)],
            max_tokens=args.max_new)
        sess = ChatSession(engine, req)
        tokens, meta = sess.run()
        print(f"user : {prompt}")
        print(f"edge0: {_display_text(tok.decode(tokens), args.show_thinking)}")
        print(f"# {len(tokens)} tokens in {meta['wall_s']}s",
              file=sys.stderr)
        return 0
    finally:
        engine.close()


def cmd_chat(args) -> int:
    from edge0 import AutoEngine

    model_dir, name = _resolve_model(args)
    if not getattr(args, "model_path", None) and (not model_dir or not os.path.isdir(model_dir)):
        raise SystemExit(
            "chat requires a checkpoint; pass it as the model argument "
            "(edge0 chat /path/to/model) or --model-dir")
    engine = AutoEngine.from_pretrained(model_dir, name=name,
                                        **_engine_kwargs(args))
    try:
        tok = engine._tok
        if tok is None:
            raise SystemExit("model has no tokenizer; cannot chat")
        if args.prompt:
            prompts = [args.prompt]
        elif not sys.stdin.isatty():
            prompts = [line.rstrip("\n") for line in sys.stdin if line.strip()]
        else:
            raise SystemExit("pass --prompt or pipe input on stdin")
        for p in prompts:
            from edge0.server.chat import ChatMessage, ChatRequest, ChatSession
            req = ChatRequest(model=name, messages=[
                ChatMessage(role="user", content=p)],
                max_tokens=args.max_new)
            sess = ChatSession(engine, req)
            diagnostics = None
            if getattr(args, "stats", False) or getattr(args, "verbose_tokens", False):
                from edge0.chat_stats import ChatStats
                diagnostics = ChatStats(engine, verbose_tokens=getattr(args, "verbose_tokens", False))
                diagnostics.emit("start")
            try:
                tokens, meta = sess.run(on_token=diagnostics.on_token if diagnostics else None,
                                        on_prompt=diagnostics.on_prompt if diagnostics else None)
            except BaseException:
                if diagnostics:
                    diagnostics.emit("interrupted")
                raise
            if diagnostics:
                diagnostics.emit("complete", meta)
            print(_display_text(tok.decode(tokens), args.show_thinking))
            print(f"# {len(tokens)} tokens in {meta['wall_s']}s", file=sys.stderr)
        return 0
    finally:
        engine.close()


def cmd_serve(args) -> int:
    from edge0 import AutoEngine
    from edge0.server import QueueServer, run_server

    model_dir, name = _resolve_model(args)
    if not getattr(args, "model_path", None) and (not model_dir or not os.path.isdir(model_dir)):
        raise SystemExit(
            "serve requires a checkpoint; pass it as the model argument "
            "(edge0 serve /path/to/model) or --model-dir")
    engine = AutoEngine.from_pretrained(model_dir, name=name,
                                        **_engine_kwargs(args))
    try:
        server = QueueServer(engine, model_name=name or engine.name)
        print(f"[edge0] serving {server.model_name} on http://{args.host}:{args.port} "
              f"(stream={'flask' if args.flask else 'stdlib'})",
              file=sys.stderr)
        run_server(server, host=args.host, port=args.port, use_flask=args.flask)
        return 0
    finally:
        engine.close()


def cmd_convert(args) -> int:
    import runpy
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve()
                           .parents[1] / "scripts"))
    runpy.run_path(
        str(__import__("pathlib").Path(__file__).resolve().parents[1]
            / "scripts" / "convert_adapters_legacy.py"),
        run_name="__main__",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="edge0", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("models", help="list registered model tiers")
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser(
        "demo",
        help="one-shot generation demo")
    p.add_argument("model", nargs="?", default=None,
                   help="tier name (edge0-35b) or checkpoint dir")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new", type=int, default=None,
                   help="max tokens to generate (default: tier config)")
    p.add_argument("--show-thinking", action="store_true",
                   help="print the model's reasoning block too")
    p.add_argument("--no-prerouter", action="store_true")
    p.add_argument("--no-lora", action="store_true")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser(
        "chat",
        help="one-shot prompt answering")
    p.add_argument("model", nargs="?", default=None,
                   help="tier name (edge0-35b) or checkpoint dir")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new", type=int, default=None,
                   help="max tokens to generate (default: tier config)")
    p.add_argument("--show-thinking", action="store_true",
                   help="print the model's reasoning block too")
    p.add_argument("--no-prerouter", action="store_true")
    p.add_argument("--no-lora", action="store_true")
    p.set_defaults(fn=cmd_chat)
    p.add_argument("--cache-mib", type=int, default=None,
                   help="GGUF packed-weight cache budget in MiB (minimum 8; default: model configuration)")
    p.add_argument("--stats", action="store_true",
                   help="log timing, cache, checkpoint reads and memory statistics to stderr")
    p.add_argument("--verbose-tokens", action="store_true",
                   help="log each prompt/generated token ID, piece and timing; includes --stats")

    p = sub.add_parser(
        "serve",
        help="run the OpenAI-compatible HTTP server: edge0 serve <model>")
    p.add_argument("model", nargs="?", default=None,
                   help="tier name (edge0-35b) or checkpoint dir")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--flask", action="store_true",
                   help="use the Flask transport (needs flask installed)")
    p.add_argument("--no-prerouter", action="store_true")
    p.add_argument("--no-lora", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("convert-adapters",
                       help="one-shot legacy npz -> safetensors migration")
    p.set_defaults(fn=cmd_convert)

    for command in ('demo', 'chat', 'serve'):
        parser = sub.choices[command]
        parser.add_argument('--model-path', help='local GGUF shard or custom safetensors directory')
        parser.add_argument('--tokenizer-path', help='local tokenizer directory for --model-path')

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
