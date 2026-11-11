#!/bin/sh
cd "/home/lech/Strata"
exec "/home/lech/Strata/.venv/bin/python" "/home/lech/Strata/serve/server.py" "--engine" "strata" "--config" "/home/lech/Strata/strata-iq3_s.json" "--port" "8080" "--open"
