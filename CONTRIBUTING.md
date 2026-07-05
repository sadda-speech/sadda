# Contributing to sadda

Howdy! 👋 If you're reading this, you're probably thinking about poking at
sadda — a next-generation phonetics / speech-science tool. Welcome! We
appreciate all constructive contributions, whether you're fixing a typo,
filing a thoughtful bug, or adding a whole new acoustic measure or
visualization.

This document is short on purpose. It's the handful of ground rules that keep
sadda coherent as it grows, plus how to actually get a change in. None of it is
meant to be scary — if in doubt, open a draft PR or an issue and we'll figure it
out together.

## House rules

These are the conventions that make sadda _sadda_ rather than a pile of loosely
related scripts. Hold to them and review will be much faster and easier.

### 1. Features ship on all three surfaces

sadda has three faces, and a user-facing capability should show up on all of
them in the same change:

- **Engine** — the Rust core (`crates/`), where the real work happens.
- **Python** — the `sadda` package, the scripting / notebook surface.
- **GUI** — the egui desktop app.

Why all three? A formant tracker that exists in the engine but not in Python,
or in Python but not the app, leaves users guessing about what sadda can actually
do — so we don't ship a capability that only half-arrives.

And each surface a feature lands on comes with **tests** — the feature isn't
done until they're in place, so the _next_ change can't quietly break it.

**Exceptions:** a surface-specific _bugfix_ (the spectrogram
mis-paints only in the GUI; a `.pyi` stub drifted; etc.) obviously only touches
that surface. Likewise, a contribution that's deliberately scoped to one surface
is welcome — just say so in the PR, and we may open follow-up issues to bring
the other surfaces level. The rule is about _new capabilities_, not about
forcing busywork onto a one-line fix.

### 2. Efficiency and latency are first-class

This is an interactive tool people use on real recordings, sometimes hours long.
Speed isn't a nice-to-have — a sluggish spectrogram or a pitch track that stalls
the UI is a bug. When you add or change something, spare a thought for:

- **Throughput** — does it scale to long files without quadratic surprises?
- **Latency** — does it keep the UI responsive (work off the UI thread, stream
  rather than load-it-all, cache where it pays)?
- **Memory** — decoding a multi-hour file shouldn't blow up RAM.

You don't need to micro-optimize everything, but a clearly wasteful approach
will get flagged. If a change trades speed for clarity on purpose, just note it.

### 3. Python: typed and tidy

The Python surface is fully type-hinted and **PEP 8**-clean, and we'd like to
keep it that way:

- **Full type hints** on public functions, methods, and the `.pyi` stubs — the
  stubs are checked for drift in the gate, so keep them honest.
- **PEP 8** formatting and naming. Match the style of the code already around
  you when in doubt.
- Phonetics notation: lowercase `f0` for fundamental frequency, uppercase `F1`,
  `F2`, etc. for formants.

### 4. Methods cite their sources

sadda is a scientific instrument. If you add a DSP or clinical measure, cite the
publication it comes from in the code, and prefer offering established
alternatives over a single house method where the literature genuinely disagrees
(e.g. multiple pitch trackers). Clinical / proprietary-origin measures are
clean-room reproductions from published descriptions — never ports of proprietary
scripts. This is for both scientific and licensing clarity.

### 5. Experimental / holdout code never ships

Perf spikes and throwaway experiments are welcome in-tree — but they must not
compile into the app or wheel. Gate them behind an **off-by-default Cargo
feature** (we use `spike`) and/or keep them in `benches/` (compiled only by
`cargo bench`, never linked into the library). A feature-gated module can still
reach engine internals; a plain default build won't see it. Keep the code green
"when run" — `just bench` enables `--features spike` and verifies any spike that
mirrors a production path still matches it. Don't leave experimental code in a
default-compiled path "temporarily"; if it's worth keeping, gate it, and if it's
not, delete it (git history is the archive).

### 6. AI assistance is welcome; AI authorship is not

Use whatever tools help you write a good contribution — including AI assistants.
But **you are the author of everything you submit, and you are responsible for
it**: the code, the commit messages, and the PR description. A tool doesn't
contribute to sadda any more than a compiler or a text editor does. Two rules
follow, and CI enforces them:

- **No AI authorship notation.** Don't credit an assistant as an author or
  co-author, and don't leave generation footers. Concretely: no
  `Co-authored-by:` (or `Assisted-by:` / `Generated-by:`) trailers naming an
  assistant — Claude, Copilot, Cursor, and the like — and no "Generated
  with …" / 🤖 footers in commit messages or PR descriptions. If a tool inserts
  these by default, strip them before you push. Commit under your own name, too
  — not an AI bot account — since a squash merge folds commit authorship into
  `Co-authored-by:` trailers.
- **No personal AI-assistant config in the repo.** Your assistant's workspace
  configuration (`.claude/`, `.cursor/`, …) is how _you_ work, not part of
  sadda — keep it local (it's gitignored).

This isn't a judgment on using AI. It's about a clean, honest record of who
stands behind each change — and that's you.

## Filing a good issue

Many contributions start as an issue. A well formed issue lets
someone act without having to interview you first. A quick checklist before you
hit submit:

- **Search first.** Someone may have already reported it; an extra detail on an
  existing issue is more useful than a duplicate.
- **For bugs, tell us three things:** what you did, what you _expected_, and what
  actually happened. A minimal set of steps — or a short script — that reproduces
  it is the fastest way for it to get solved.
- **Say where you are.** sadda version, OS, Python version, and which surface
  (engine / Python / GUI) — the same bug can look very different across the three.
- **Bring the evidence.** Paste the full error / traceback for crashes; for GUI
  glitches, what you clicked and a screenshot if you can. If one specific
  recording triggers it, a small, shareable clip helps enormously.
- **For feature requests, lead with the _why_.** Describe what you're trying to
  accomplish, not only the solution you already have in mind — the use case helps
  us weigh it, and there may be a better way.

Rule of thumb: an actionable issue is one a stranger could pick up and reproduce
(or act on) without needing to ask you a single follow-up question. You don't
have to be perfect — a rough issue beats a silent bug every time — but a little
detail goes a long way.

## How to get a change in

We work in branches and pull requests — **no direct commits to `main`**. `main`
is protected and only moves through reviewed PRs with green CI. This keeps the
history clean and reviewable, which matters more every day as more people show
up.

1. **Branch off `main`.** Name it for what it does — `feat/burg-formants`,
   `fix/spectrogram-repaint`, `docs/quickstart-typo`. Whatever's clear.
2. **Make your change**, tests included (see house rule #1).
3. **Run the gate locally** before you push:

   ```sh
   just gate
   ```

   This mirrors CI exactly — `cargo fmt`, `clippy`, the build, the full Rust +
   Python test suites, and the type-stub drift check. A green `just gate` means
   a green CI; running it locally saves you the round-trip.

4. **Open a pull request into `main`.** Describe what changed and why. Draft PRs
   are very welcome if you want early eyes.
5. **CI runs the same gate** on your PR. It needs to be green to merge — a
   broken commit can't reach `main`, which is exactly the point.
6. A maintainer reviews, you iterate if needed, and in it goes. 🎉

**Keep your branch merge-commit-free.** `main` enforces a linear history, so PRs
land by squash or rebase, not a merge commit. When your branch falls behind,
**rebase it onto `main`** (`git rebase main`) rather than merging `main` into it
(`git merge main`). Merging `main` in bakes a merge commit into your branch,
which then blocks GitHub's rebase-merge option — leaving squash as the only way
in, so a branch with a carefully staged commit series gets flattened to one.
Rebasing keeps that history landable and `main` clean.

## Recognition

If you land significant contributions, we'd love to register you with the team
— we're happy to invite meaningful contributors into the
[`sadda-speech`](https://github.com/sadda-speech) GitHub organization.

## Questions?

Open an issue, start a discussion, or leave a comment on a draft PR. There are no
silly questions. If you are a prospective first-time contributor and are
wondering how to get involved, have a look at the posted issues, and feel free
to ask questions in the issues if you need guidance or anything is unclear.

Thanks for helping make sadda better. 🙇
