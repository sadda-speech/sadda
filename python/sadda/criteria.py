"""S2 criteria-engine helpers: run a *structured* or *python-escape* criterion
to materialize proposals onto its preview tier (``"<target> (auto)"``).

``structured`` criteria are delegated to the engine (``Project.run_criterion``).
A ``python`` criterion's ``body`` must define a function
``criterion(proj, bundle_id)`` that returns an iterable of proposals, each a
tuple ``(start, end_or_None, label_or_None)`` — ``end=None`` denotes a point.
Shorter tuples are accepted: ``(start,)`` and ``(start, end)`` default the
missing fields.

This API is provisional.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sadda._stability import provisional

if TYPE_CHECKING:
    from sadda import Project

__all__ = ["run_criterion"]


def _normalize(proposal: Any) -> tuple[float, float | None, str | None]:
    """Coerces a user-returned proposal into ``(start, end_or_None, label)``."""
    p = tuple(proposal)
    if not p:
        raise ValueError("a proposal must have at least a start time")
    start = float(p[0])
    end = float(p[1]) if len(p) > 1 and p[1] is not None else None
    label = p[2] if len(p) > 2 else None
    return (start, end, label)


@provisional
def run_criterion(proj: Project, criterion_id: int, bundle_id: int) -> int:
    """Runs criterion ``criterion_id`` against ``bundle_id`` and (re)writes its
    proposals onto the preview tier, replacing any prior ones. Returns the
    proposal count.

    Structured criteria run in the engine; python criteria are executed here:
    the ``body`` is ``exec``'d and its ``criterion(proj, bundle_id)`` function
    is called to produce proposals.
    """
    crit = proj.get_criterion(criterion_id)
    if crit is None:
        raise ValueError(f"no criterion with id {criterion_id}")
    if crit.kind == "structured":
        return proj.run_criterion(criterion_id, bundle_id)
    if crit.kind != "python":
        raise ValueError(f"unknown criterion kind {crit.kind!r}")

    namespace: dict[str, Any] = {}
    exec(crit.body, namespace)  # noqa: S102 - user-authored criterion body
    fn = namespace.get("criterion")
    if not callable(fn):
        raise ValueError(
            "a python criterion body must define a `criterion(proj, bundle_id)` function"
        )
    proposals = [_normalize(p) for p in fn(proj, bundle_id)]
    # Trace the python run exactly like a structured one: record the
    # criterion_run first, then stamp each proposal with its provenance link.
    run_id = proj.record_criterion_run(criterion_id, bundle_id)
    return proj.set_proposals(bundle_id, crit.target_tier, proposals, run_id)
