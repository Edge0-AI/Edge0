#!/usr/bin/env python3
"""my_load.py URL OUT.json - par_load.py (this harness) with PAR-A armrun's workload (W2 1k prompt, 128 out, c=8,16, 2 rounds), as armrun --loadcmd."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import par_load
sys.exit(par_load.main([sys.argv[1], "--workload", "w2", "--conc", "8,16", "--rounds", "2", "--max-tokens", "128", "--warm-requests", "2", "--out", sys.argv[2]]))
