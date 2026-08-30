"""Shared model utilities for neuroprobe_eval models."""

from numbers import Integral


def _as_int(value, *, context: str) -> int:
    """Return ``value`` as an int while rejecting bool and non-integral types."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{context} must contain only int values, got {value!r}.")
    return int(value)


def to_int_list(value, default, allow_empty=False):
    """Convert scalar/list-like config value to a list of ints."""
    if value is None:
        return [_as_int(v, context="default") for v in list(default)]
    if hasattr(value, "__iter__") and not isinstance(value, str):
        out = [_as_int(v, context="value") for v in list(value)]
        if out:
            return out
        if allow_empty:
            return []
        return [_as_int(v, context="default") for v in list(default)]
    if isinstance(value, bool):
        raise TypeError(f"value must be an int or iterable of ints, got {value!r}.")
    if isinstance(value, Integral):
        return [int(value)]
    raise TypeError(
        "value must be None, an int, or an iterable of ints; "
        f"got {type(value).__name__}."
    )


def coerce_int_list(value, default, allow_empty=False):
    """Legacy-compatible int-list coercion used by older model configs.

    This intentionally preserves the older permissive behavior for configs that
    may contain floats or other int-castable values.
    """
    if value is None:
        return list(default)
    if hasattr(value, "__iter__") and not isinstance(value, str):
        out = [int(v) for v in list(value)]
        if out:
            return out
        return [] if allow_empty else list(default)
    if isinstance(value, (int, float)):
        return [int(value)]
    return list(default)
