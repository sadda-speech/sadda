"""Tests for the stability-tier decorators and warning categories."""

from __future__ import annotations

import warnings

import pytest

import sadda
from sadda._stability import (
    ExperimentalAPIWarning,
    ProvisionalAPIWarning,
    SaddaWarning,
    _tier_registry,
    _warned,
    experimental,
    get_stability,
    provisional,
    stable,
)


@pytest.fixture(autouse=True)
def reset_warn_state() -> None:
    """Each test starts with a clean once-set so its assertions are
    independent of test order."""
    _warned.clear()
    yield
    _warned.clear()


def test_phase0_surface_is_importable_and_tiered() -> None:
    assert callable(sadda.load_wav)
    assert callable(sadda.f0)
    assert isinstance(sadda.Audio, type)
    for sym in (sadda.load_wav, sadda.f0, sadda.Audio):
        assert get_stability(sym) == "stable", sym
    # Version + schema are plain value constants, not tiered callables.
    assert isinstance(sadda.__version__, str)
    assert isinstance(sadda.SCHEMA_VERSION, int)


def test_warning_classes_inherit_from_user_warning() -> None:
    assert issubclass(SaddaWarning, UserWarning)
    assert issubclass(ProvisionalAPIWarning, SaddaWarning)
    assert issubclass(ExperimentalAPIWarning, SaddaWarning)


def test_stable_does_not_warn() -> None:
    @stable
    def fn() -> int:
        return 42

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SaddaWarning)
        fn()
        fn()
    assert caught == []
    assert fn() == 42
    assert get_stability(fn) == "stable"


def test_provisional_warns_once_then_forwards() -> None:
    @provisional
    def fn(x: int) -> int:
        return x + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SaddaWarning)
        assert fn(1) == 2
        assert fn(2) == 3
        assert fn(3) == 4
    assert len(caught) == 1
    assert issubclass(caught[0].category, ProvisionalAPIWarning)
    assert "provisional" in str(caught[0].message).lower()
    assert get_stability(fn) == "provisional"


def test_experimental_warns_once() -> None:
    @experimental
    def fn() -> str:
        return "ok"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SaddaWarning)
        fn()
        fn()
    assert len(caught) == 1
    assert issubclass(caught[0].category, ExperimentalAPIWarning)


def test_user_can_silence_all_tiers_via_base_class() -> None:
    @provisional
    def p() -> None: ...

    @experimental
    def e() -> None: ...

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", SaddaWarning)
        p()
        e()
    assert caught == []


def test_provisional_class_warns_on_first_init_only() -> None:
    @provisional
    class C:
        def __init__(self, x: int) -> None:
            self.x = x

        def method(self) -> int:
            return self.x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SaddaWarning)
        a = C(1)
        b = C(2)
        assert a.method() == 1
        assert b.method() == 2
    assert len(caught) == 1
    assert issubclass(caught[0].category, ProvisionalAPIWarning)
    assert get_stability(C) == "provisional"


def test_isinstance_check_does_not_warn() -> None:
    @provisional
    class C:
        def __init__(self) -> None: ...

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SaddaWarning)
        # Type-only references — no instantiation, no warning.
        _ = C
        _ = isinstance(object(), C)
    assert caught == []


def test_decorator_preserves_docstring_and_name() -> None:
    @provisional
    def documented(x: int) -> int:
        """Adds one."""
        return x + 1

    assert documented.__doc__ == "Adds one."
    assert documented.__name__ == "documented"


def test_pyo3_symbols_register_tier_even_without_attribute_set() -> None:
    # Native PyO3 functions / classes don't accept arbitrary attribute writes,
    # so the registry-based lookup is the only path that works for them.
    # Confirm that path is being exercised (and not silently broken).
    key = f"{sadda._native.load_wav.__module__}.{sadda._native.load_wav.__qualname__}"
    assert _tier_registry.get(key) == "stable"
