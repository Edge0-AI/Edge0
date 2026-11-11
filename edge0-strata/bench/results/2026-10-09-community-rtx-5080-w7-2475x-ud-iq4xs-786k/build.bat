@echo off
rem Build pulito di v0.1.41 (fb58e0d, 08/10/2026) in un worktree nuovo C:\strata-lab\pr\strata-0.1.41.
rem Stessi flag di build-0.1.39.bat. La 0.1.40.1 quotidiana non viene toccata. Build da zero: 10-20 minuti.
setlocal
set WT=C:\strata-lab\pr\strata-0.1.41
set WANT=fb58e0dbc8399662c0e47c76578c6e878b14f6cf
where git >nul 2>&1 || (echo ERRORE: git non e' nel PATH & exit /b 1)
rem Solo il tag che serve: upstream ha riscritto la storia e i tag vecchi (v0.1.0-v0.1.39) ora puntano a commit
rem diversi dai tuoi locali; un fetch --tags li rifiuta ("would clobber existing tag") e va bene cosi'.
git -C C:\Strata-main fetch origin refs/tags/v0.1.41:refs/tags/v0.1.41 || exit /b 1
if not exist "%WT%\.git" (
  git -C C:\Strata-main worktree add --detach "%WT%" v0.1.41 || exit /b 1
)
for /f %%h in ('git -C "%WT%" rev-parse HEAD') do set HEAD=%%h
echo HEAD del worktree: %HEAD%
if not "%HEAD%"=="%WANT%" (
  echo ERRORE: atteso %WANT%
  exit /b 1
)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
C:\Users\Enky\anaconda3\Scripts\cmake.EXE -G Ninja -DCMAKE_MAKE_PROGRAM=C:\Users\Enky\anaconda3\Scripts\ninja.EXE -S "%WT%" -B "%WT%\build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120 "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.EXE" -DSTRATA_GGML_DIR=C:\Strata-main\third_party\llama.cpp || exit /b 1
C:\Users\Enky\anaconda3\Scripts\cmake.EXE --build "%WT%\build" --target strata -j 20 && goto ok
echo   (la build si e' fermata - riprovo una volta)
C:\Users\Enky\anaconda3\Scripts\cmake.EXE --build "%WT%\build" --target strata -j 20 || exit /b 1
:ok
echo.
echo BUILD OK: %WT%\build\strata.exe
pause
