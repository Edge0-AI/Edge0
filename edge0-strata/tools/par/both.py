#!/usr/bin/env python3
"""both.py URL OUT.json - armrun --loadcmd: the sustained closed-loop reference (this harness, W2 1k/128, c from BOTH_CONC, 6 rounds per client)
followed by PAR-A's synchronized-burst loader (env PA_LOAD, 2 measured rounds after a warm round).  Both read in one server start."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
import par_load
url, out = sys.argv[1], sys.argv[2]
conc = os.environ.get("BOTH_CONC", "8,16,32")
print("### SUSTAINED (par_load.py, closed loop, rounds=6)", flush=True)
par_load.main([url, "--workload", "w2", "--conc", conc, "--rounds", "6", "--max-tokens", "128", "--warm-requests", "2",
               "--nonce", "sus", "--out", out + ".sustained.json"])
print("### BURST (PAR-A par_load.py, synchronized waves)", flush=True)
sys.stdout.flush()
subprocess.run([sys.executable, os.environ["PA_LOAD"], url, "--conc", conc, "--rounds", "2", "--prompt-tokens", "1000", "--max-tokens", "128",
                "--out", out + ".burst.json"])
