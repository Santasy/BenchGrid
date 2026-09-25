"""
Shared, feature-agnostic helpers of BenchGrid.

Currently holds the grid input model (``Grid`` + ``expand_grid``), used by the
unified ``plan`` model to map combinations to commands. Intentionally small:
the frame de-dup policy moves logic here only once a second consumer exists.
"""

from .grid import Grid, expand_grid

__all__ = ["Grid", "expand_grid"]