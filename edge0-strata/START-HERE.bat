@echo off
rem Strata for Windows: the first run installs everything and starts the model; later runs just start it.
rem Needs only an NVIDIA or AMD graphics driver. Python is installed for your user account if it is missing (no admin needed).
setlocal
rem Python reads and writes text as UTF-8, not in the Windows code page (cp1251, cp866, ...): a folder or user name with Cyrillic or other letters stays intact (#3202).
set "PYTHONUTF8=1"
title Strata
cd /d "%~dp0"
rem A bundled Python runtime (the python\ folder of the terminal archive) is used when it is there: nothing to find,
rem install or download. SETUP.bat and UPDATE.bat run this file, so they use it too.
if exist "python\python.exe" goto bundled
if exist ".venv\Scripts\python.exe" goto run

call :findpy
if defined PY goto venv
echo.
echo  Python 3.10 or newer is not installed. Installing Python 3.12 for your user account ...
where winget >nul 2>nul
if errorlevel 1 goto pyorg
winget install -e --id Python.Python.3.12 --scope user --silent --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
call :findpy
if defined PY goto venv
:pyorg
echo  Downloading the Python installer from python.org ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe -OutFile \"$env:TEMP\strata-python-setup.exe\""
if exist "%TEMP%\strata-python-setup.exe" "%TEMP%\strata-python-setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
call :findpy
if defined PY goto venv
echo.
echo  Python could not be installed automatically.
echo  Install 64-bit Python 3.12 from https://www.python.org/downloads/ ("Add python.exe to PATH"),
echo  then double-click START-HERE.bat again.
pause
exit /b 1

:venv
rem a private environment inside this folder, so nothing is installed into the system Python
%PY% -m venv .venv
if exist ".venv\Scripts\python.exe" goto run
echo  Could not create the Python environment in .venv
pause
exit /b 1

:bundled
set "PYTHONNOUSERSITE=1"
set "PYTHONHOME="
set "PYTHONPATH="
"python\python.exe" setup.py %*
if errorlevel 1 pause
exit /b

:run
".venv\Scripts\python.exe" setup.py %*
if errorlevel 1 pause
exit /b

:findpy
rem the py launcher first, then python on PATH (not the Microsoft Store stub), then the usual per-user folders
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3" & goto :eof
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python" & goto :eof
for %%V in (313 312 311 310) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"" & goto :eof
goto :eof
