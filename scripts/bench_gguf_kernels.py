"""Measure cold compile+dispatch and warm execution using pinned C fixtures."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='/tmp/gguf-kernel-timings.json')
    args = p.parse_args()
    import mlx.core as mx
    from edge0.checkpoints.encodings import ENCODINGS
    from edge0.backends.mlx.gguf_packed import matmul
    fixture = np.load(Path(__file__).resolve().parents[1]/'tests/fixtures/gguf_quant_reference.npz')
    report = dict(device=mx.device_info(), note='Cold time includes compilation and first dispatch; compilation is estimated as cold minus median warm time.', encodings={})
    for kind, encoding in ENCODINGS.items():
        w = mx.array(fixture[f'raw_{kind}'].reshape(-1))
        x = mx.ones((3, 8*encoding.block_elements), dtype=mx.float32)
        ids = mx.array([0, 1, 0], dtype=mx.uint32)
        mx.eval(w, x, ids)
        durations = []
        for i in range(11):
            start = time.perf_counter()
            y = matmul([w, w], kind, x, ids, 4)
            mx.eval(y)
            durations.append(time.perf_counter()-start)
        warm = float(np.median(durations[1:]))
        report['encodings'][encoding.name] = dict(cold_compile_dispatch_s=durations[0], warm_s=warm,
                                                 estimated_compile_overhead_s=max(0, durations[0]-warm))
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
