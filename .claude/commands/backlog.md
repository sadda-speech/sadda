---
description: Capture a development idea to BACKLOG.md without derailing current work
argument-hint: <idea to queue>
---
Capture this development idea into the queue, fast and without losing the current thread: $ARGUMENTS

Do exactly this:
1. Append it to `BACKLOG.md` at the repo root, under the `## Inbox` section, as a single checklist line:
   `- [ ] <concise restatement of the idea> — _added <today's date, YYYY-MM-DD>_`
   - Create `BACKLOG.md` and/or the `## Inbox` heading if they are missing (keep the file's existing structure otherwise).
   - One line, preserving the user's intent lightly tightened. If the idea clearly extends an already-listed grouped item, add a sub-bullet under that item instead of a new Inbox line.
   - If the user's text includes obvious context worth keeping (a file, a symptom, a constraint), keep it terse in the same line — do not expand into a design.
2. Reply with ONE short confirmation line only, e.g. `Queued: <short title>.` — no analysis, no questions, no design, no offer to start it.
3. Immediately resume whatever we were doing before this capture, as if uninterrupted.

This is a low-friction inbox. Triage/grooming happens only when the user explicitly asks (e.g. "groom the backlog"). Never start work on a captured idea from this command.
