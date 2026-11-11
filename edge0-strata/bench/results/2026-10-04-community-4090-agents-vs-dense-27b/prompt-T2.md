You are working on a ticket in the Aphotic-Hypr repo (Quickshell/QML desktop shell). Work ONLY inside the git
worktree WT (cd there first). Do not edit <Aphotic-Hypr checkout> or <user config>, do not launch, reload or restart the live
shell or Hyprland, do not push. Commit your finished work on the current branch (no attribution trailers).
Read CLAUDE.md and AGENTS.md in the worktree and follow them. Qt/Quickshell API docs: rg in <local Qt docs>.

Ticket B0ABG9E - "settings: profile picture in the settings menu":
Put a profile picture in the settings menu. The settings surface carries no user identity at all right now, so
this is a new element rather than a change to an existing one.
Constraints:
- Local first. No cloud, and no new external dependency. The image lives under the user's own config directory.
- Settings is Quickshell QML. Use the existing file dialog helper and load the result with a file URL.
- The shell already has a cache-busting pattern in the live Wallpapers.qml for an image that gets overwritten in
  place. Follow it, or a re-picked avatar shows the old one.
- Persist the choice through the existing settings service and add the key to the same schema, so it round-trips
  through the normal config sync.
- Clearing the avatar has to return the header to its current layout without leaving a gap.
Open question: header only, or the panel as well (decide, and say why).
Done when an avatar can be picked and cleared, it survives a reboot, and a missing or unreadable image leaves the
settings menu as it looks now.

Verify statically (qmllint on the files you touch, any repo checks the docs name). End with a short report: files
changed, how you verified it (exact commands and results), what you could not verify, open questions.
