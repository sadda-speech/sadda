"""sadda.recipe — reproducibility for analyses (F1).

A context manager that links every processing_run row written inside
the block to a named recipe_run row in the project's corpus.db. On
clean exit, emits ``<project>/recipes/<name>.py`` — a runnable
Python script that re-issues the captured operations.

Capture scope in v1: the three engine methods that currently INSERT
``processing_run`` rows — ``Project.import_textgrid``,
``Project.import_eaf``, and ``LiveSession.commit``. Pure-DSP calls
(``sadda.dsp.f0`` etc.) are not captured; the user's own Python
script is the orchestration record. See the 2026-05-22 F1 DEVLOG
entry for the full design.

Typical usage::

    with sadda.recipe.record(proj, name="vowel_analysis_v1") as rec:
        proj.import_textgrid(Path("phones.TextGrid"), bundle_id)
        proj.import_eaf(Path("annotations.eaf"), bundle_id)
    # → corpus.db has a recipe_run row + linked processing_runs
    # → <project>/recipes/vowel_analysis_v1.py exists

    sadda.recipe.list(proj)        # → [Recipe(...)]
    sadda.recipe.script_path(proj, "vowel_analysis_v1")
                                   # → str path to the generated .py

The generated ``.py`` is a regular Python script — run it with
``python <name>.py [project_path]``. No ``sadda.recipe.replay()`` in
v1; the script IS the replay mechanism.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from sadda import _native
from sadda._stability import provisional

__all__ = ["Recipe", "list", "get", "record", "script_path"]


Recipe = _native.recipe.Recipe


class _RecordContext:
    """Context manager returned by :func:`record`. Opens a recipe at
    ``__enter__`` and finalises (status + generated .py) on
    ``__exit__``. The recipe id is exposed as ``self.recipe_id``."""

    def __init__(
        self,
        project: Any,
        name: str,
        parameters: Optional[dict] = None,
    ) -> None:
        self._project = project
        self._name = name
        self._parameters = parameters
        self.recipe_id: Optional[int] = None

    def __enter__(self) -> "_RecordContext":
        params_json = (
            json.dumps(self._parameters, sort_keys=True)
            if self._parameters is not None
            else None
        )
        self.recipe_id = _native.recipe.start(
            self._project, self._name, params_json
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.recipe_id is None:
            return False
        if exc_type is None:
            _native.recipe.end(self._project, self.recipe_id, "ok", None)
            # Best-effort .py generation. If it fails, we still want
            # the recipe row to be marked ok — the SQL provenance is
            # the source of truth; the script is a convenience.
            try:
                _native.recipe.generate_script(self._project, self.recipe_id)
            except Exception:
                pass
        else:
            _native.recipe.end(
                self._project, self.recipe_id, "error", repr(exc)
            )
        return False  # don't suppress


@provisional
def record(
    project: Any,
    name: str,
    parameters: Optional[dict] = None,
) -> _RecordContext:
    """Opens a recipe block. Returns a context manager — use it in a
    ``with`` statement.

    :param project: a :class:`sadda.Project`.
    :param name: unique recipe name within the project. Recording the
        same name twice errors (UNIQUE constraint on
        ``recipe_run.name``).
    :param parameters: optional dict of user-supplied metadata,
        serialised to JSON and recorded in ``recipe_run.parameters``.
        Not interpreted by sadda; available to your own tooling via
        :func:`get`.
    """
    return _RecordContext(project, name, parameters)


@provisional
def list(project: Any) -> "list[Recipe]":  # noqa: A001 (shadows builtins.list)
    """Lists every recipe in the project in id order."""
    return _native.recipe.list_recipes(project)


@provisional
def get(project: Any, name: str) -> Recipe:
    """Fetches a single recipe by name. Raises if the name isn't found."""
    return _native.recipe.get(project, name)


@provisional
def script_path(project: Any, name: str) -> str:
    """Returns the conventional path to a recipe's `.py` script,
    whether or not the file exists. Use this to discover where the
    generator would write."""
    return _native.recipe.script_path(project, name)
