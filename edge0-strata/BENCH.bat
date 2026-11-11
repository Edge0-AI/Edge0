@echo off
rem Benchmark this Strata install the way it is used and write a report you can share (docs/COMMUNITY_BENCHMARKS.md).
rem It starts the model with your settings (or uses the one already running), takes about 10 minutes and changes nothing.
rem Options: --clients 4 (several requests at once), --long (a 32K prompt), --compare <older results folder>.
rem This is the quick speed check. The standard benchmark (docs/BENCHMARK_PROTOCOL.md), whose results other people can compare with, is `strata bench run`.
setlocal
title Strata - benchmark
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo  Strata is not set up here yet: run START-HERE.bat first.
  pause
  exit /b 1
)
echo  Quick speed check. For results other people can compare with (the standard benchmark, a few minutes), run: .venv\Scripts\python.exe strata.py bench run
echo.
".venv\Scripts\python.exe" tools\strata_bench.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
