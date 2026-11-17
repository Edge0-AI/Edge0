#!/bin/sh
# Update Strata without starting the model (#475): the newest code (git pull, when this folder is a git clone), then
# what ./setup.sh does before a start - the engine (a new ready-made one when this version needs it, or compiled
# again when its source changed), the Python packages, each installed model's settings and draft subset. The model
# files are not touched. Start the model later with ./setup.sh. Options are passed on to setup.py. The bundled Python
# (python/ beside this file) is used when there is one: ./setup.sh, which this file runs, picks it.
cd "$(dirname "$0")" || exit 1
# all of it in a function, read before it runs: the git pull below can change this very file
main() {
  if [ -e .git ]; then
    if ! command -v git >/dev/null 2>&1; then
      echo "This folder is a git clone, but git is not installed: install it (sudo apt install git) or run"
      echo "\"git pull\" here yourself, then run ./update.sh again."
      exit 1
    fi
    echo "Getting the newest Strata (git pull) ..."
    if ! git pull --ff-only; then
      # The history of the repository was rewritten on 2026-10-06 (#1276): a clone made before that has no commit in
      # common with origin/main, so a fast-forward can never work. Say so, and move it over when nothing is lost.
      if [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" != "true" ] && ! git merge-base HEAD origin/main >/dev/null 2>&1; then
        echo
        echo "This clone is on the repository's old history (cleaned up on 2026-10-06): it has no commit in common"
        echo "with origin/main, so it cannot be updated by git pull."
        if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
          echo "You have local changes to tracked files here, so nothing was touched. To move to the new history"
          echo "yourself (your untracked files, models and settings stay where they are):"
          echo "  git branch pre-cleanup-backup && git stash push   # keeps the old commits and your edits"
          echo "  git checkout -B main origin/main"
          echo "then run ./update.sh again."
          exit 1
        fi
        bk=pre-cleanup-backup
        if git rev-parse --verify -q "refs/heads/$bk" >/dev/null 2>&1; then
          bk="pre-cleanup-backup-$(date +%Y%m%d%H%M%S)"
        fi
        branch=$(git symbolic-ref --short -q HEAD || echo main)
        if git branch "$bk" HEAD && git checkout -q -B "$branch" origin/main; then
          echo "Moved to the new history. Your old commits are kept in the branch $bk."
          echo "Untracked files (models, settings, the engine) were not touched."
        else
          # the backup branch made a moment ago would make the by-hand `git branch pre-cleanup-backup` below fail
          git branch -D "$bk" >/dev/null 2>&1
          echo "Could not move to the new history (the reason is above): nothing was updated. By hand:"
          echo "  git branch pre-cleanup-backup"
          echo "  git checkout -B main origin/main"
          exit 1
        fi
      else
        echo
        echo "git pull did not succeed (the reason is above): nothing was updated. Files you changed here can stop it:"
        echo "\"git status\" lists them."
        exit 1
      fi
    fi
  else
    echo "This copy of Strata was not made with git, so it cannot fetch new files itself. Download the newest one:"
    echo "  https://github.com/Niko1221/Strata/archive/refs/heads/main.zip"
    echo "unzip it anywhere and run ./setup.sh (or ./update.sh) in it: it finds the model files in Strata-data and"
    echo "sets itself up the same way - nothing big is downloaded again."
    echo "Checking this copy's engine and settings meanwhile ..."
  fi
  exec sh ./setup.sh --update "$@"
}
main "$@"
