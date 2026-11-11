@echo off
rem umaprobe.exe for gfx1151 (HIP, ROCm 10.2 wheels + MSVC headers), into %OUT% (default C:\wdsv41s\umaprobe.exe).  ROCM_VENV as in build_hip_win.bat.
setlocal
if not defined ROCM_VENV set "ROCM_VENV=%~dp0..\..\..\.rocm-win"
if not defined OUT set "OUT=C:\wdsv41s\umaprobe.exe"
for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "ROCM=%%R"
for /f "delims=" %%V in ('"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VS=%%V"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
"%ROCM%\lib\llvm\bin\clang++.exe" -x hip -O2 --offload-arch=gfx1151 --rocm-path="%ROCM%" --rocm-device-lib-path="%ROCM%\lib\llvm\amdgcn\bitcode" -I"%ROCM%\include" "%~dp0umaprobe.cpp" -o "%OUT%" -L"%ROCM%\lib" -lamdhip64 -ldxgi -lversion -lpsapi -Wno-unused-command-line-argument
echo BUILD-EXIT %errorlevel%
endlocal
