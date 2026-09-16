"""Category definitions for the submitted eight-category benchmark."""

from __future__ import annotations


REAL3D_AD_CATEGORIES = (
    "candybar",
    "diamond",
    "fish",
    "toffees",
)

MVTEC_3D_AD_CATEGORIES = (
    "bagel",
    "carrot",
    "peach",
    "potato",
)

BENCHMARK_CATEGORIES = REAL3D_AD_CATEGORIES + MVTEC_3D_AD_CATEGORIES

_MVTEC_3D_AD_CATEGORY_SET = frozenset(MVTEC_3D_AD_CATEGORIES)
_BENCHMARK_CATEGORY_SET = frozenset(BENCHMARK_CATEGORIES)


def is_mvtec_category(category: str) -> bool:
    """Return whether *category* is in the benchmark's MVTec 3D-AD subset."""

    return str(category).strip().lower() in _MVTEC_3D_AD_CATEGORY_SET


def validate_benchmark_category(category: str) -> str:
    """Normalize and validate one final-benchmark category name."""

    normalized = str(category).strip().lower()
    if normalized not in _BENCHMARK_CATEGORY_SET:
        expected = ", ".join(BENCHMARK_CATEGORIES)
        raise ValueError(
            f"Unknown benchmark category {category!r}; expected one of: "
            f"{expected}."
        )
    return normalized


__all__ = [
    "REAL3D_AD_CATEGORIES",
    "MVTEC_3D_AD_CATEGORIES",
    "BENCHMARK_CATEGORIES",
    "is_mvtec_category",
    "validate_benchmark_category",
]
