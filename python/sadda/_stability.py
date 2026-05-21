"""Stability-tier decorators and warning categories.

Every public sadda symbol carries one of three stability tiers, applied via
:func:`stable`, :func:`provisional`, or :func:`experimental`. The first time
a non-stable symbol is used in a given process, a warning is emitted; the
``warnings`` machinery treats it like any other warning, so users can
silence it with ``warnings.simplefilter("ignore", SaddaWarning)``.

See the 2026-05-18 Python API surface DEVLOG entry and the 2026-05-21 A2
entry for the design.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, TypeVar

__all__ = [
    "SaddaWarning",
    "ProvisionalAPIWarning",
    "ExperimentalAPIWarning",
    "stable",
    "provisional",
    "experimental",
    "get_stability",
]


class SaddaWarning(UserWarning):
    """Base class for sadda's stability warnings.

    Subclassing ``UserWarning`` (not ``DeprecationWarning``) keeps the
    warnings visible by default; Python silences ``DeprecationWarning``
    in non-``__main__`` modules, which would hide the very signal these
    decorators are designed to surface.
    """


class ProvisionalAPIWarning(SaddaWarning):
    """First use of a PROVISIONAL API.

    Provisional APIs may break in minor versions after a deprecation cycle.
    They are intentionally exposed for early users; expect them to firm up
    over the next few releases.
    """


class ExperimentalAPIWarning(SaddaWarning):
    """First use of an EXPERIMENTAL API.

    Experimental APIs may break in any release without notice. They live
    under ``sadda.experimental.*`` (or are individually marked) and exist so
    end users can opt in to early feedback on in-development surfaces.
    """


F = TypeVar("F", bound=Callable[..., Any])

_warned: set[str] = set()
# Sidecar registry: maps qualified name → tier. PyO3 builtins don't accept
# arbitrary attribute writes, so we keep a fallback registry here. The
# attribute is set best-effort on targets that allow it (pure-Python
# functions, classes, wrappers); ``get_stability`` looks at both.
_tier_registry: dict[str, str] = {}


def _qualified_name(obj: Any) -> str:
    module = getattr(obj, "__module__", "") or ""
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", repr(obj))
    return f"{module}.{qualname}" if module else qualname


def _tag(target: Any, tier: str) -> None:
    _tier_registry[_qualified_name(target)] = tier
    try:
        target.__sadda_stability__ = tier
    except (AttributeError, TypeError):
        # PyO3 builtin / C class — attribute mutation isn't allowed.
        # The registry still records the tier for `get_stability`.
        pass


def get_stability(obj: Any) -> str | None:
    """Returns the stability tier (``"stable"``, ``"provisional"``,
    ``"experimental"``) for a decorated symbol, or ``None`` if the symbol
    carries no tier marker.
    """
    tier = getattr(obj, "__sadda_stability__", None)
    if tier is not None:
        return tier
    return _tier_registry.get(_qualified_name(obj))


def _warn_once(key: str, category: type[Warning], message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(message, category=category, stacklevel=3)


def _decorate_callable(
    target: F, category: type[Warning] | None, tier: str, kind_message: str
) -> F:
    _tag(target, tier)
    if category is None:
        # @stable: no wrapping; just the tier tag.
        return target

    key = _qualified_name(target)
    message = f"{key} is a {kind_message} sadda API and may change in future releases"

    @functools.wraps(target)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _warn_once(key, category, message)
        return target(*args, **kwargs)

    _tag(wrapper, tier)
    wrapper.__wrapped__ = target  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def _decorate_class(
    target: type, category: type[Warning] | None, tier: str, kind_message: str
) -> type:
    _tag(target, tier)
    if category is None:
        return target

    key = _qualified_name(target)
    message = f"{key} is a {kind_message} sadda API and may change in future releases"
    original_init = target.__init__

    @functools.wraps(original_init)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _warn_once(key, category, message)
        original_init(self, *args, **kwargs)

    try:
        target.__init__ = wrapped_init  # type: ignore[method-assign]
    except (AttributeError, TypeError):
        # PyO3 #[pyclass] types have a read-only __init__; fall back to
        # wrapping __new__ if possible. For now we just leave it untagged
        # at the wrapper level — the registry entry from _tag still works.
        pass
    return target


def _make_decorator(
    category: type[Warning] | None, tier: str, kind_message: str
) -> Callable[[Any], Any]:
    def decorator(obj: Any) -> Any:
        if inspect.isclass(obj):
            return _decorate_class(obj, category, tier, kind_message)
        return _decorate_callable(obj, category, tier, kind_message)

    decorator.__name__ = tier
    decorator.__doc__ = f"Marks the decorated function or class as {tier.upper()}."
    return decorator


stable = _make_decorator(None, "stable", "stable")
provisional = _make_decorator(ProvisionalAPIWarning, "provisional", "provisional")
experimental = _make_decorator(ExperimentalAPIWarning, "experimental", "experimental")
