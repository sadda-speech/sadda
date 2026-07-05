#!/usr/bin/env bash
#
# Contribution-policy enforcement (see CONTRIBUTING.md, house rule #6:
# "AI assistance is welcome; AI authorship is not").
#
# Contributors own their changes; AI agents are tools, not authors. This fails
# a PR that:
#   - carries an AI co-author/assist trailer or generation footer in a commit
#     message or the PR description,
#   - has a commit *authored or committed by* an AI-agent identity (this is the
#     one that matters for squash merges: GitHub rolls commit authorship into
#     `Co-authored-by:` trailers on the squash commit — which lands on main
#     WITHOUT re-running this check — so we catch the AI-authored commit here,
#     before the squash), or
#   - adds an AI assistant's personal workspace config to the tree.
#
# It inspects commit messages, authorship, the PR body, and changed paths —
# never file contents — so prose that merely discusses these tools (e.g. a
# DEVLOG entry about removing trailers) is not flagged.
#
# The AGENTS list is a heuristic blocklist, extended in one place below. A rare
# false positive (e.g. a human whose name/email collides) is a maintainer
# override, not a reason to weaken it.
#
# Inputs (from the workflow):
#   BASE_SHA  merge-base commit on the target branch
#   HEAD_SHA  PR head commit
#   PR_BODY   the pull-request description (passed via env, never interpolated)
set -euo pipefail

BASE="${BASE_SHA:?BASE_SHA is required}"
HEAD="${HEAD_SHA:?HEAD_SHA is required}"
PR_BODY="${PR_BODY:-}"

# Known AI coding assistants / agents / models. Case-insensitive. Word
# boundaries guard the short, human-name-ish tokens (codex, cody, grok, jules).
AGENTS='claude|anthropic|openai|chatgpt|gpt-[0-9]|\bcodex\b|copilot|cursor|gemini|\bbard\b|\bjules\b|google-labs-jules|devin|\bcognition\b|aider|codeium|windsurf|sourcegraph|\bcody\b|tabnine|codewhisperer|amazon[ -]?q\b|q-developer|replit|ghostwriter|sweep-?ai|qodo|codiumai|codestral|deepseek|phind|perplexity|\bgrok\b|bolt\.new|lovable|v0\.dev|codellama|code-llama|blackbox-?ai|supermaven|augmentcode|augment-code|tabbyml|zencoder|continue\.dev|llama-?index|autogpt|smol-?agents'

# AI-agent authorship / assist trailers (a trailer *line*, not a passing
# mention): Co-authored-by / Assisted-by / Generated-by / Created-by / Authored-by.
TRAILER="^[[:space:]]*(co-authored-by|assisted-by|generated-by|created-by|authored-by):[[:space:]].*($AGENTS)"

# Generation footers ("Generated with/by <agent>") or a bare robot-emoji marker.
FOOTER="(generated (with|by)[[:space:]].*($AGENTS))|🤖"

# Identity scan uses only *distinctive* handles + bot-account markers + vendor
# domains — NOT the human-name-ish tokens (claude, cody, jules, bard, gemini,
# grok), so a contributor literally named Cody or Claude isn't flagged. Real AI
# committers for those vendors show up as bot accounts / vendor emails anyway
# (e.g. `Claude <noreply@anthropic.com>`, `gemini-code-assist[bot]`).
IDENTITY_AGENTS='anthropic|openai|chatgpt|copilot|cursor|devin|aider|codeium|windsurf|sourcegraph|tabnine|codewhisperer|q-developer|replit|ghostwriter|sweep-?ai|qodo|codiumai|codestral|deepseek|perplexity|bolt\.new|v0\.dev|blackbox-?ai|supermaven|augmentcode|augment-code|tabbyml|zencoder|continue\.dev|google-labs-jules|autogpt|smol-?agents'
BOT='\[bot\]|-ai-integration|swe-agent|noreply@anthropic\.com|@cursor\.com'
IDENTITY="($IDENTITY_AGENTS)|($BOT)"

# Personal AI-assistant workspace config that shouldn't live in the repo.
AI_PATHS='^(\.claude/|\.cursor/|\.aider|\.windsurf|\.continue/|\.codeium/|\.github/copilot-)'

fail=0
flag() { printf '::error::%s\n' "$1" >&2; fail=1; }

# 1. Commit messages in the PR range. `git log -z` NUL-separates commits; each
#    record is "<sha>\n<full message>".
while IFS= read -r -d '' record; do
  sha="${record%%$'\n'*}"
  msg="${record#*$'\n'}"
  if printf '%s' "$msg" | grep -qiE "$TRAILER"; then
    flag "Commit ${sha:0:12}: AI co-author/assist trailer. You are the author — remove it (CONTRIBUTING.md #6)."
  fi
  if printf '%s' "$msg" | grep -qiE "$FOOTER"; then
    flag "Commit ${sha:0:12}: AI generation footer. Remove it (CONTRIBUTING.md #6)."
  fi
done < <(git log -z --format='%H%n%B' "$BASE..$HEAD")

# 2. Author / committer identity — the squash-merge vector. A commit authored by
#    an AI identity becomes a Co-authored-by trailer on the squash commit, so
#    catch it here on the pre-squash commit.
while IFS=$'\x1f' read -r sha who; do
  if printf '%s' "$who" | grep -qiE "$IDENTITY"; then
    flag "Commit ${sha:0:12}: authored/committed by an AI identity ($who). Commit under your own name (CONTRIBUTING.md #6)."
  fi
done < <(git log --format='%H%x1f%an <%ae> | %cn <%ce>' "$BASE..$HEAD")

# 3. PR description.
if printf '%s' "$PR_BODY" | grep -qiE "$TRAILER|$FOOTER"; then
  flag "PR description contains AI authorship/generation notation. Remove it (CONTRIBUTING.md #6)."
fi

# 4. Changed paths — no personal AI-assistant config *added* to the tree.
#    `--diff-filter=d` excludes deletions, so *removing* such config (e.g. the
#    commit that gitignores .claude/) is not itself flagged.
if changed=$(git diff --name-only --diff-filter=d "$BASE..$HEAD" | grep -iE "$AI_PATHS"); then
  flag "PR adds AI-assistant workspace config: $(printf '%s' "$changed" | tr '\n' ' '). Keep it local/gitignored (CONTRIBUTING.md #6)."
fi

if [ "$fail" -ne 0 ]; then
  echo "" >&2
  echo "Contribution policy check failed — see CONTRIBUTING.md, house rule #6." >&2
  echo "You may use AI tools; you just can't credit them as authors. The record is yours." >&2
  exit 1
fi
echo "Contribution policy: OK — no AI authorship notation found."
