"""Ligand-based pharmacophore profiling with RDKit."""

from pharmprof.alerts import flag_alerts
from pharmprof.metrics import bedroc, enrichment_factor, pr_auc, roc_auc, screening_metrics
from pharmprof.patterns import Pattern, PatternError, PatternSet, builtin_patterns, load_patterns
from pharmprof.profiler import profile
from pharmprof.screen import apply_filter, pharm2d_similarity
from pharmprof.stats import compare_classes, pattern_correlation, pattern_summary

__version__ = "0.1.0"

__all__ = [
    "Pattern",
    "PatternError",
    "PatternSet",
    "apply_filter",
    "bedroc",
    "builtin_patterns",
    "compare_classes",
    "enrichment_factor",
    "flag_alerts",
    "load_patterns",
    "pattern_correlation",
    "pattern_summary",
    "pharm2d_similarity",
    "pr_auc",
    "profile",
    "roc_auc",
    "screening_metrics",
    "__version__",
]
