"""Analysis module exports."""

from seam.analysis.aggregator import ResultAggregator
from seam.analysis.plotting import (
    plot_contamination_propagation,
    plot_memory_collapse,
    plot_performance_comparison,
)

__all__ = [
    "ResultAggregator",
    "plot_contamination_propagation",
    "plot_memory_collapse",
    "plot_performance_comparison",
]
