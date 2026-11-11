@echo off
rem Update Strata without starting the model (#475): the newest code (git pull, when this folder is a git clone), then
rem what START-HERE.bat does before a start - the engine (a new one when this version needs it), the Python packages,
rem each installed model's settings and draft subset. The model files are not touched. Start the model later with
rem START-HERE.bat. Options are passed on to setup.py. The bundled Python (python\ beside this file) is used when
rem there is one: START-HERE.bat, which this file runs, picks it.
setlocal
title Strata - update
cd /d "%~dp0"
rem All of it in one block: cmd reads a .bat file while it runs it, and the git pull can change this very file.
(
  if exist ".git" (
    where git >nul 2>nul || (
      echo  This folder is a git clone, but git is not on PATH: install Git for Windows ^(winget install Git.Git^)
      echo  or run "git pull" here yourself, then run UPDATE.bat again.
      pause
      exit /b 1
    )
    echo  Getting the newest Strata ^(git pull^) ...
    git pull --ff-only
    if errorlevel 1 (
      rem The history was rewritten on 2026-10-06, see #1276: a clone from before has no commit in common with origin/main.
      set "REWR="
      git merge-base HEAD origin/main >nul 2>nul
      if errorlevel 1 set "REWR=1"
      git rev-parse --is-shallow-repository 2>nul | findstr /x true >nul && set "REWR="
      if defined REWR (
        echo.
        echo  This clone is on the repository's old history ^(cleaned up on 2026-10-06^): it has no commit in common
        echo  with origin/main, so it cannot be updated by git pull.
        set "DIRTY="
        for /f "delims=" %%i in ('git status --porcelain --untracked-files^=no') do set "DIRTY=1"
        if defined DIRTY (
          echo  You have local changes to tracked files here, so nothing was touched. To move to the new history
          echo  yourself ^(your untracked files, models and settings stay where they are^):
          echo    git branch pre-cleanup-backup
          echo    git stash push
          echo    git checkout -B main origin/main
          echo  then run UPDATE.bat again.
          pause
          exit /b 1
        )
        set "BK=pre-cleanup-backup"
        git rev-parse --verify -q refs/heads/pre-cleanup-backup >nul 2>nul && set "BK=pre-cleanup-backup-%RANDOM%"
        set "BR=main"
        for /f "delims=" %%i in ('git symbolic-ref --short -q HEAD 2^>nul') do set "BR=%%i"
        call git branch %%BK%% HEAD
        if errorlevel 1 (
          echo  Could not make the backup branch: nothing was updated.
          pause
          exit /b 1
        )
        call git checkout -q -B %%BR%% origin/main
        if errorlevel 1 (
          call git branch -D %%BK%% >nul 2>nul
          echo.
          echo  Could not move to the new history ^(the reason is above^): nothing was updated. By hand:
          echo    git checkout -B main origin/main
          pause
          exit /b 1
        )
        call echo  Moved to the new history. Your old commits are kept in the branch %%BK%%.
        echo  Untracked files ^(models, settings, the engine^) were not touched.
      ) else (
        echo.
        echo  git pull did not succeed ^(the reason is above^): nothing was updated. Files you changed here can stop it:
        echo  "git status" lists them.
        pause
        exit /b 1
      )
    )
  ) else (
    echo  This copy of Strata was not made with git, so it cannot fetch new files itself. Download the newest one:
    echo    https://github.com/Niko1221/Strata/archive/refs/heads/main.zip
    echo  unzip it anywhere and run START-HERE.bat ^(or UPDATE.bat^) in it: it finds the model files in Strata-data
    echo  and sets itself up the same way - nothing big is downloaded again.
    echo  Checking this copy's engine and settings meanwhile ...
  )
  call "%~dp0START-HERE.bat" --update %*
  if not errorlevel 1 pause
  exit /b
)
