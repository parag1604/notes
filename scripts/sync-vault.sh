#!/usr/bin/env bash
# Sync the Obsidian vault (source of truth) into Quartz's content/ folder.
#
#   scripts/sync-vault.sh            # preview the changes (dry run)
#   scripts/sync-vault.sh --apply    # actually copy
#
# - Vault Readme.md becomes content/index.md (the site homepage).
# - CLAUDE.md, _meta/, .obsidian/, .git/ and .DS_Store are never copied.
# - Files deleted from the vault are deleted from content/ too.
# Afterwards: review `git status`, then commit and push on the v4 branch.
set -euo pipefail

VAULT="${VAULT:-$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/NotesVault}"
SITE="$(cd "$(dirname "$0")/.." && pwd)"
CONTENT="$SITE/content"

DRY="--dry-run"
[[ "${1:-}" == "--apply" ]] && DRY=""

[[ -f "$VAULT/Readme.md" ]] || { echo "Vault not found at: $VAULT" >&2; exit 1; }

rsync -a -m --no-perms --delete $DRY --itemize-changes \
  --exclude '.git/' \
  --exclude '.obsidian/' \
  --exclude '.DS_Store' \
  --exclude '.gitignore' \
  --exclude '_meta/' \
  --exclude 'CLAUDE.md' \
  --exclude 'Readme.md' \
  --exclude '/index.md' \
  "$VAULT/" "$CONTENT/"

if [[ -z "$DRY" ]]; then
  cp "$VAULT/Readme.md" "$CONTENT/index.md"
  # iCloud files arrive as 755; normalise so git doesn't see mode changes
  find "$CONTENT" -path "$CONTENT/.obsidian" -prune -o -type f -exec chmod 644 {} +
  echo "Synced. Readme.md -> content/index.md"
  echo "Next: cd \"$SITE\" && git status && npx quartz build --serve   (preview)"
else
  cmp -s "$VAULT/Readme.md" "$CONTENT/index.md" || echo ">f.st...... index.md  (from Readme.md)"
  echo "(dry run: re-run with --apply to copy)"
fi
