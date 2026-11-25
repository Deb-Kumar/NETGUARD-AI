"""NETGUARD AI command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from src.data.eda import run_eda_pipeline
from src.data.loader import (
    DatasetLoadError,
    iter_csv_chunks,
    get_feature_metadata_map,
    load_training_and_testing_sets,
    profile_dataset,
)
from src.data.validator import (
    PREDEFINED_FEATURE_COLUMNS,
    extract_features_and_target,
    validate_train_test_compatibility,
)
from src.evaluation import (
    build_comparison_table,
    evaluate_binary_model,
    generate_binary_evaluation_plots,
    save_evaluation_results,
)
from src.models import MODEL_SPECS, load_model_artifact
from src.training import (
    SELECTION_STRATEGIES,
    run_grid_search_tuning,
    run_stratified_cv,
    select_and_finalize_best_model,
    train_all_binary_models,
    train_and_evaluate_multiclass,
)
from src.utils.config import (
    APP_VERSION,
    CV_FOLDS,
    PROJECT_NAME,
    PROJECT_TITLE,
    ensure_directories,
    get_environment_summary,
    get_paths,
)
from src.utils.logger import configure_logging

CORE_PACKAGES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "plotly",
    "streamlit",
    "joblib",
    "pytest",
)


def collect_dependency_versions() -> dict[str, str | None]:
    """Return installed versions of required project packages, or None if missing."""
    versions: dict[str, str | None] = {}
    for pkg in CORE_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def run_check_env() -> int:
    """Inspect runtime environment, installed packages, and dataset paths."""
    paths = ensure_directories(get_paths(Path(__file__).resolve().parent))
    summary = get_environment_summary(paths)
    summary["dependencies"] = collect_dependency_versions()
    print(json.dumps(summary, indent=2))
    missing = [k for k, v in summary["dependencies"].items() if v is None]
    if missing:
        print(f"Missing required packages: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


def run_inspect_data(
    *,
    downcast: bool = False,
    include_raw_parts: bool = False,
    save_report: bool = False,
) -> int:
    """Load, profile, and validate UNSW-NB15 datasets without altering raw files."""
    paths = ensure_directories(get_paths(Path(__file__).resolve().parent))
    try:
        train_df, test_df = load_training_and_testing_sets(
            downcast_numeric=downcast,
            paths=paths,
        )
        feature_map = get_feature_metadata_map(paths=paths)
    except DatasetLoadError as exc:
        print(f"Dataset loading error: {exc}", file=sys.stderr)
        return 1

    train_profile = profile_dataset(
        train_df,
        name="UNSW_NB15_training-set",
        feature_columns=PREDEFINED_FEATURE_COLUMNS,
    )
    test_profile = profile_dataset(
        test_df,
        name="UNSW_NB15_testing-set",
        feature_columns=PREDEFINED_FEATURE_COLUMNS,
    )
    validation = validate_train_test_compatibility(train_df, test_df)

    report: dict[str, Any] = {
        "feature_descriptions_mapped": len(
            [col for col in train_df.columns if col in feature_map]
        ),
        "training_set": train_profile,
        "testing_set": test_profile,
        "validation": validation.to_dict(),
    }

    if include_raw_parts:
        raw_parts_summary: dict[str, dict[str, int]] = {}
        for filename, part_path in paths.raw_part_paths.items():
            if part_path is None:
                continue
            rows = 0
            chunks = 0
            for chunk in iter_csv_chunks(part_path, chunksize=250_000, has_header=False):
                rows += len(chunk)
                chunks += 1
            raw_parts_summary[filename] = {"rows": rows, "chunks": chunks}
        report["raw_parts_summary"] = raw_parts_summary

    if save_report:
        out_path = paths.reports_experiments_dir / "dataset_inspection_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["saved_report_path"] = str(out_path)

    print(json.dumps(report, indent=2))
    return 0 if validation.is_valid else 1


def run_eda(*, no_plots: bool = False) -> int:
    """Execute full Exploratory Data Analysis and write reports/graphs/."""
    paths = ensure_directories(get_paths(Path(__file__).resolve().parent))
    try:
        eda_summary = run_eda_pipeline(paths=paths, generate_plots=not no_plots)
    except DatasetLoadError as exc:
        print(f"Dataset loading error during EDA: {exc}", file=sys.stderr)
        return 1

    compact_output = {
        "summary_path": eda_summary["summary_path"],
        "training_records": eda_summary["training_profile"]["num_records"],
        "testing_records": eda_summary["testing_profile"]["num_records"],
        "duplicate_and_conflict_analysis": eda_summary["duplicate_and_conflict_analysis"],
        "top_5_correlated_features": dict(
            list(eda_summary["correlation_analysis"]["target_correlations"].items())[:5]
        ),
        "generated_figures": eda_summary["generated_figures"],
        "sampling_notes": eda_summary["sampling_notes"],
    }
    print(json.dumps(compact_output, indent=2))
    return 0


def run_train(*, skip_imbalance_study: bool = False) -> int:
    """Train all three binary classifiers on the UNSW-NB15 training set."""
    paths = ensure_directories(get_paths(Path(__file__).resolve().parent))
    summary = train_all_binary_models(
        run_imbalance_study=not skip_imbalance_study,
        paths=paths,
    )
    print(
        json.dumps(
            {
                "summary_path": summary["summary_path"],
                "preprocessing_config_path": summary["preprocessing_config_path"],
                "models": {
                    k: {
                        "training_duration_seconds": v["training_duration_seconds"],
                        "artifact_path": v["artifact_path"],
                    }
                    for k, v in summary["models"].items()
                },
            },
            indent=2,
        )
    )
    return 0


def run_cv(*, n_splits: int = CV_FOLDS, max_samples: int | None = None) -> int:
    """Run Stratified K-Fold cross-validation strictly on training data."""
    paths = ensure_directories(get_paths(Path(__file__).resolve().parent))
    cv_summary = run_stratified_cv(
        n_splits=n_splits,
        max_samples=max_samples,
        paths=paths,
    )
    print(
        json.dumps(
            {
                "results_path": cv_summary["results_path"],
                "dataset_scope": cv_summary["dataset_scope"],
                "comparison_table": cv_summary["comparison_table"],
            },
            indent=2,
        )
    )
    return 0


def run_tune(
    *,
    cv_folds: int = CV_FOLDS,
    scoring: str = "f1",
    n_jobs: int = 1,
    max_tuning_samples: int | None = None,
) -> int:
    """Run GridSearchCV hyperparameter tuning on training data only."""
    paths = ensure_directories(get_paths(Path(__file__).resolve().parent))
    tuning_summary = run_grid_search_tuning(
        cv_folds=cv_folds,
        scoring=scoring,
        n_jobs=n_jobs,
        max_tuning_samples=max_tuning_samples,
        paths=paths,
    )
    print(
        json.dumps(
            {
                "results_path": tuning_summary["results_path"],
                "search_scope": tuning_summary["search_scope"],
                "models": {
                    k: {
                        "best_params": v["best_params"],
                        "best_cv_score": v["best_cv_score"],
                        "best_cv_metrics": v["best_cv_metrics"],
                        "tuning_duration_seconds": v["tuning_duration_seconds"],
                    }
                    for k, v in tuning_summary["models"].items()
                },
            },
            indent=2,
        )
    )
    return 0


def run_evaluate(*, no_plots: bool = False) -> int:
    """Evaluate all saved binary models on the untouched UNSW-NB15 testing set."""
    paths = ensure_directories(get_paths(Path(__file__).resolve().parent))
    _, test_df = load_training_and_testing_sets(paths=paths)
    X_test, y_test = extract_features_and_target(test_df, task="binary")

    evaluations: dict[str, dict[str, Any]] = {}
    for key, spec in MODEL_SPECS.items():
        artifact = paths.binary_models_dir / f"{key}.joblib"
        if not artifact.is_file():
            continue
        model = load_model_artifact(artifact, paths=paths)
        evaluations[key] = evaluate_binary_model(
            model,
            X_test,
            y_test,
            model_key=key,
            model_name=spec.display_name,
            split_name="untouched_test",
        )

    if not evaluations:
        print("No trained binary models found. Run 'python main.py train' first.", file=sys.stderr)
        return 1

    eval_path = save_evaluation_results(evaluations, paths=paths)
    plot_paths = (
        {}
        if no_plots
        else generate_binary_evaluation_plots(evaluations, paths=paths)
    )
    comparison = build_comparison_table(list(evaluations.values())).to_dict(orient="records")
    print(
        json.dumps(
            {
                "evaluation_results_path": str(eval_path),
                "plots": plot_paths,
                "comparison_table": comparison,
            },
            indent=2,
        )
    )
    return 0


def run_select_best(*, strategy: str = "cybersecurity_composite") -> int:
    """Select best model from validation scores and perform final test evaluation."""
    paths = ensure_directories(get_paths(Path(__file__).resolve().parent))
    metadata_payload = select_and_finalize_best_model(
        strategy=strategy,  # type: ignore[arg-type]
        paths=paths,
        generate_plots=True,
    )
    print(
        json.dumps(
            {
                "metadata_path": metadata_payload["metadata_path"],
                "selected_model_key": metadata_payload["selected_model_key"],
                "selected_model_name": metadata_payload["selected_model_name"],
                "selection_strategy": metadata_payload["selection_strategy"],
                "selected_validation_metrics": metadata_payload["selected_validation_metrics"],
                "final_test_metrics": metadata_payload["final_test_evaluation"]["metrics"],
                "all_models_test_comparison": metadata_payload["all_models_test_comparison"],
            },
            indent=2,
        )
    )
    return 0


def run_train_multiclass(*, no_plot: bool = False) -> int:
    """Train and evaluate the independent multiclass attack_cat classifier."""
    paths = ensure_directories(get_paths(Path(__file__).resolve().parent))
    metadata = train_and_evaluate_multiclass(paths=paths, generate_plot=not no_plot)
    print(
        json.dumps(
            {
                "metadata_path": metadata["metadata_path"],
                "selected_model_key": metadata["selected_model_key"],
                "selected_model_name": metadata["selected_model_name"],
                "discovered_classes": metadata["discovered_classes"],
                "selected_metrics": metadata["models"][metadata["selected_model_key"]]["metrics"],
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for NETGUARD AI."""
    parser = argparse.ArgumentParser(
        prog="netguard",
        description=f"{PROJECT_NAME} - {PROJECT_TITLE}",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROJECT_NAME} {APP_VERSION}",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "check-env",
        help="Inspect Python environment, dependencies, and dataset availability.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-data",
        help="Load, profile, and validate UNSW-NB15 datasets.",
    )
    inspect_parser.add_argument("--downcast", action="store_true")
    inspect_parser.add_argument("--include-raw-parts", action="store_true")
    inspect_parser.add_argument("--save-report", action="store_true")

    eda_parser = subparsers.add_parser(
        "run-eda",
        help="Run Exploratory Data Analysis and generate visualizations in reports/.",
    )
    eda_parser.add_argument("--no-plots", action="store_true")

    train_parser = subparsers.add_parser(
        "train",
        help="Train Logistic Regression, Decision Tree, and Random Forest binary models.",
    )
    train_parser.add_argument("--skip-imbalance-study", action="store_true")

    cv_parser = subparsers.add_parser(
        "cross-validate",
        help="Run 5-fold Stratified CV strictly on training data.",
    )
    cv_parser.add_argument("--n-splits", type=int, default=CV_FOLDS)
    cv_parser.add_argument("--max-samples", type=int, default=None)

    tune_parser = subparsers.add_parser(
        "tune",
        help="Run GridSearchCV hyperparameter tuning on training data only.",
    )
    tune_parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    tune_parser.add_argument("--scoring", type=str, default="f1")
    tune_parser.add_argument("--n-jobs", type=int, default=1)
    tune_parser.add_argument("--max-tuning-samples", type=int, default=None)

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained models on the untouched UNSW-NB15 testing set.",
    )
    eval_parser.add_argument("--no-plots", action="store_true")

    select_parser = subparsers.add_parser(
        "select-best",
        help="Select best model via CV validation metrics and save best_model.joblib.",
    )
    select_parser.add_argument(
        "--strategy",
        choices=list(SELECTION_STRATEGIES.keys()),
        default="cybersecurity_composite",
    )

    mc_parser = subparsers.add_parser(
        "train-multiclass",
        help="Train and evaluate independent multiclass attack_cat classifier.",
    )
    mc_parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the NETGUARD AI CLI."""
    logger = configure_logging("netguard", log_to_file=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command:
        logger.info("Executing NETGUARD AI CLI command: %s", args.command)

    if args.command == "check-env":
        return run_check_env()
    if args.command == "inspect-data":
        return run_inspect_data(
            downcast=args.downcast,
            include_raw_parts=args.include_raw_parts,
            save_report=args.save_report,
        )
    if args.command == "run-eda":
        return run_eda(no_plots=args.no_plots)
    if args.command == "train":
        return run_train(skip_imbalance_study=args.skip_imbalance_study)
    if args.command == "cross-validate":
        return run_cv(n_splits=args.n_splits, max_samples=args.max_samples)
    if args.command == "tune":
        return run_tune(
            cv_folds=args.cv_folds,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            max_tuning_samples=args.max_tuning_samples,
        )
    if args.command == "evaluate":
        return run_evaluate(no_plots=args.no_plots)
    if args.command == "select-best":
        return run_select_best(strategy=args.strategy)
    if args.command == "train-multiclass":
        return run_train_multiclass(no_plot=args.no_plot)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
