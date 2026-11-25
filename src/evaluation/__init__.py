"""Model evaluation metrics, comparison reporting, and visualization plots."""

from src.evaluation.evaluate import (
    build_comparison_table,
    evaluate_binary_model,
    evaluate_multiclass_model,
    save_evaluation_results,
)
from src.evaluation.plots import (
    generate_binary_evaluation_plots,
    plot_binary_confusion_matrices,
    plot_model_comparison_bars,
    plot_multiclass_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
)

__all__ = [
    "build_comparison_table",
    "evaluate_binary_model",
    "evaluate_multiclass_model",
    "generate_binary_evaluation_plots",
    "plot_binary_confusion_matrices",
    "plot_model_comparison_bars",
    "plot_multiclass_confusion_matrix",
    "plot_pr_curves",
    "plot_roc_curves",
    "save_evaluation_results",
]
