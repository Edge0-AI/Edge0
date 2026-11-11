@echo off
title Strata IQ2_XS (STRATA_PLE_BATCH=0)
cd /d "D:\AI\Strata\Strata-main\Strata-main"
set STRATA_PLE_BATCH=0
echo Starting Strata with STRATA_PLE_BATCH=0
"D:\AI\Strata\Strata-main\Strata-main\.venv\Scripts\python.exe" "D:\AI\Strata\Strata-main\Strata-main\serve\server.py" "--engine" "strata" "--config" "D:\AI\Strata\Strata-main\Strata-main\strata-iq2_xs.json" "--port" "8080" "--open"
if errorlevel 1 pause
