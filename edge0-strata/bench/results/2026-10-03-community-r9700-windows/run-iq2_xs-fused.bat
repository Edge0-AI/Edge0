@echo off
title Strata IQ2_XS (STRATA_PF_FUSED=1)
cd /d "D:\AI\Strata\Strata-main\Strata-main"
set STRATA_PF_FUSED=1
echo Starting Strata with STRATA_PF_FUSED=1
"D:\AI\Strata\Strata-main\Strata-main\.venv\Scripts\python.exe" "D:\AI\Strata\Strata-main\Strata-main\serve\server.py" "--engine" "strata" "--config" "D:\AI\Strata\Strata-main\Strata-main\strata-iq2_xs.json" "--port" "8080" "--open"
if errorlevel 1 pause
