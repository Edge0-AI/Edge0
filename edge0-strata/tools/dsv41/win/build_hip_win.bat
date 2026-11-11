@echo off
rem strata-dsv41 + the kernel parity tests for AMD (HIP) on Windows, ROCm 10.2 (TheRock wheels), default gfx1151 (Strix Halo).
rem
rem   tools\dsv41\win\build_hip_win.bat [targets...]       (default: strata-dsv41 ds4_iq_test)
rem
rem Same toolchain as tools\hip\build_windows.bat: the ROCm venv, MSVC's headers and linker from vcvars, ROCm's clang for host and
rem device code.  Settings (all optional):
rem   STRATA_HIP_ARCHS   gfx1151 (also gfx1100;gfx1201 ...)      ROCM_VENV   <repo>\.rocm-win
rem   BUILD_DIR          <repo>\build-dsv41-hip                   STRATA_GGML_DIR   a llama.cpp checkout at the pinned commit
rem   JOBS               parallel compile jobs (default 6)
setlocal EnableDelayedExpansion
for %%I in ("%~dp0..\..\..") do set "SRC=%%~fI"
if not defined STRATA_HIP_ARCHS set "STRATA_HIP_ARCHS=gfx1151"
if not defined ROCM_VENV set "ROCM_VENV=%SRC%\.rocm-win"
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-dsv41-hip"
if not defined JOBS set "JOBS=6"
set "TARGETS=%*"
if "%TARGETS%"=="" set "TARGETS=strata-dsv41 ds4_iq_test"
for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "ROCM=%%R"
if not exist "%ROCM%\lib\llvm\bin\clang++.exe" (echo ROCm has no compiler in "%ROCM%" & exit /b 1)
set "ROCM_F=%ROCM:\=/%"
set "BITCODE=%ROCM_F%/lib/llvm/amdgcn/bitcode"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS="
for /f "delims=" %%V in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VS=%%V"
if not defined VS (echo Visual Studio Build Tools with the C++ workload are needed & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
set "GGML="
if defined STRATA_GGML_DIR set "GGML=-DSTRATA_GGML_DIR=%STRATA_GGML_DIR:\=/%"
if not exist "%BUILD_DIR%\build.ninja" (
  cmake -G Ninja -S "%SRC%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON ^
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_HIP_ARCHITECTURES=%STRATA_HIP_ARCHS%" ^
    "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
    "-DCMAKE_HIP_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" "-DCMAKE_HIP_COMPILER_ROCM_ROOT=%ROCM_F%" ^
    "-DCMAKE_PREFIX_PATH=%ROCM_F%" "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_F% --rocm-device-lib-path=%BITCODE%" ^
    %GGML% || exit /b 1
)
cmake --build "%BUILD_DIR%" -j %JOBS% --target %TARGETS% -- -k 0
echo BUILD-EXIT %errorlevel%
endlocal
