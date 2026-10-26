#!/usr/bin/env python3
"""Train daily XGBoost models for the six meteorological targets.

Uses prepared station-day records and saves dated Booster models,
scalers and training summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("xgboost_training")


@dataclass(frozen=True)
class Stage:
    name: str
    target: str
    features: tuple[str, ...]
    max_depth: int
    learning_rate: float
    min_child_weight: int
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    rounds: int

    def params(self) -> dict[str, Any]:
        """Original parameter dictionary, without introducing new defaults."""
        return {
            "objective": "reg:squarederror",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "eval_metric": "rmse",
        }


STAGES: tuple[Stage, ...] = (
    Stage("Tavg", "MeanTEM", ("alt", "Lon", "Lat"), 4, 0.01, 6, 0.8, 0.8, 1.0, 5.0, 2500),
    Stage("Tmax", "HighTEM", ("alt", "Lon", "Lat", "MeanTEM"), 5, 0.01, 6, 0.8, 0.8, 1.0, 5.0, 3000),
    Stage("Tmin", "LowTEM", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM"), 6, 0.01, 6, 0.8, 0.8, 1.0, 5.0, 3000),
    Stage("Ps", "MeanPress", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM", "LowTEM"), 6, 0.01, 8, 0.85, 0.85, 1.0, 5.0, 3000),
    Stage("RH", "MeanRHU", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM", "LowTEM", "MeanPress"), 6, 0.015, 8, 0.85, 0.85, 1.0, 4.0, 3300),
    Stage("SD", "Suntime", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM", "LowTEM", "MeanPress", "MeanRHU"), 6, 0.015, 8, 0.8, 0.8, 2.0, 4.0, 3300),
)
STAGE_BY_NAME = {stage.name: stage for stage in STAGES}


@dataclass(frozen=True)
class TrainingSettings:
    validation_fraction: float = 0.1
    split_random_state: int = 42
    early_stopping_rounds: int = 150
    eval_metric_name: str = "test"
    missing_value_indicator_for_audit: int = 32766


def read_config(path: Path) -> tuple[Path, TrainingSettings]:
    """The ORIGINAL six XGBoost scripts all use one shared prepared CSV folder."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    raw_path = Path(cfg["input_directory"]).expanduser()
    input_dir = raw_path.resolve() if raw_path.is_absolute() else (path.parent / raw_path).resolve()
    settings = TrainingSettings(**cfg.get("training_settings", {}))
    if not (0 < settings.validation_fraction < 1):
        raise ValueError("validation_fraction must be between zero and one")
    if settings.early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be positive")
    if settings.eval_metric_name != "test":
        raise ValueError("The original scripts use the evaluation-set label 'test'.")
    return input_dir, settings


def read_day(path: Path, stage: Stage, settings: TrainingSettings) -> tuple[np.ndarray, np.ndarray, int, dict[str, int]]:
    """Read ALL rows, matching original XGBoost training (no target filtering).

    Unlike the MLP scripts, originals read every row of a shared Suntime-stage
    CSV and pass the raw predictors and targets to StandardScaler and XGBoost.
    Do not silently exclude station rows or replace 32766 missing sentinels.
    """
    frame = pd.read_csv(path)
    columns = [*stage.features, stage.target]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns for {stage.name}: {missing}")
    # Strict checks prevent accidental non-numeric inputs without changing valid data.
    converted = frame[columns].apply(pd.to_numeric, errors="raise")
    if len(converted) < 12:
        raise ValueError(f"{path}: too few rows ({len(converted)}) for a 90/10 split")
    predictors = converted.loc[:, list(stage.features)].to_numpy(dtype=np.float64, copy=True)
    targets = converted.loc[:, stage.target].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(targets).all():
        raise ValueError(f"{path}: target contains NaN/Inf; original XGBoost training requires valid labels")
    # StandardScaler and DMatrix can accept NaN predictors; preserve original handling.
    if np.isinf(predictors).any():
        raise ValueError(f"{path}: Inf predictors cannot be used by the original pipeline")
    audit = {
        "target_sentinel_rows": int(np.sum(targets == settings.missing_value_indicator_for_audit)),
        "predictor_sentinel_cells": int(np.sum(predictors == settings.missing_value_indicator_for_audit)),
        "predictor_nan_cells": int(np.isnan(predictors).sum()),
    }
    if any(audit.values()):
        LOG.warning("%s %s: input contains suspect sentinel/NaN values: %s; no rows were filtered", stage.name, path.name, audit)
    return predictors, targets, len(frame), audit


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def output_files(root: Path, stage: Stage, date_stamp: str) -> dict[str, Path]:
    folder = root / stage.name / date_stamp
    return {
        "folder": folder,
        "booster": folder / f"complete_model_{stage.target}_checkpoint-{date_stamp}.model",
        "scaler": folder / f"{stage.target}_scaler_{date_stamp}.save",
        "metadata": folder / "training_metadata.json",
    }


def metrics(y_true: np.ndarray, y_hat: np.ndarray, prefix: str) -> dict[str, float]:
    pred = np.asarray(y_hat).reshape(-1)
    return {
        f"{prefix}_ME": float(np.mean(pred - y_true)),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, pred)),
        f"{prefix}_MSE": float(mean_squared_error(y_true, pred)),
        f"{prefix}_R2": float(r2_score(y_true, pred)),
    }


def train_day(stage: Stage, day: pd.Timestamp, csv_path: Path, root: Path,
              settings: TrainingSettings, overwrite: bool = False) -> dict[str, Any]:
    """Perform original daily XGBoost fit and original best-iteration inference.

    Native xgboost.train with validation-based early stopping; the saved booster
    includes all built rounds and records best_iteration as in the original.
    """
    import xgboost as xgb
    stamp = day.strftime("%Y%m%d")
    paths = output_files(root, stage, stamp)
    expected = {
        "source_input_sha256": sha256_file(csv_path),
        "stage": stage.name,
        "target_column": stage.target,
        "feature_columns": list(stage.features),
        "xgboost_params": stage.params(),
        "num_boost_round": stage.rounds,
        "training_settings": asdict(settings),
    }
    if paths["metadata"].exists() and not overwrite:
        previous = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (paths["booster"].is_file() and paths["scaler"].is_file()
            and all(previous.get(key) == value for key, value in expected.items())):
            LOG.info("Skipping completed %s %s", stage.name, stamp)
            return previous
        raise RuntimeError(f"Partial or mismatched existing output for {stage.name} {stamp}; verify it or pass --overwrite")
    if (paths["booster"].exists() or paths["scaler"].exists()) and not overwrite:
        raise RuntimeError(f"Existing incomplete files at {paths['folder']}; verify or pass --overwrite")

    x, y, total, audit = read_day(csv_path, stage, settings)
    # Exact original order: fit scaler on all rows, THEN random split.
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)
    x_train, x_val, y_train, y_val = train_test_split(
        x_std, y, test_size=settings.validation_fraction,
        random_state=settings.split_random_state,
    )
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    booster = xgb.train(
        params=stage.params(),
        dtrain=dtrain,
        num_boost_round=stage.rounds,
        evals=[(dtrain, "train"), (dval, "test")],
        early_stopping_rounds=settings.early_stopping_rounds,
        verbose_eval=False,
    )
    paths["folder"].mkdir(parents=True, exist_ok=True)
    # Keep the historical '.model' filename; serialization depends on XGBoost version.
    booster.save_model(str(paths["booster"]))
    joblib.dump(scaler, paths["scaler"])

    # Native Booster.predict(dval), not model sliced to best_iteration. Preserves
    # original code and XGBoost version-dependent prediction semantics.
    yhat_train = booster.predict(dtrain)
    yhat_val = booster.predict(dval)
    result: dict[str, Any] = {
        **expected,
        "date": stamp,
        "source_csv": csv_path.name,
        "all_input_rows": int(total),
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "source_warnings": audit,
        "best_iteration_zero_based": int(booster.best_iteration),
        "best_score": float(booster.best_score),
        "num_boosted_rounds_in_saved_model": int(booster.num_boosted_rounds()),
        "scaler_fit_scope": "all_input_rows_before_split (original implementation)",
        "model_file": paths["booster"].name,
        "model_file_format_note": ".model extension; encoding is XGBoost-version-dependent",
        "xgboost_version": xgb.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        **metrics(y_train, yhat_train, "train"),
        **metrics(y_val, yhat_val, "validation"),
    }
    paths["metadata"].write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("Trained %s %s: n=%s, best_iteration=%s", stage.name, stamp, len(y), booster.best_iteration)
    return result


def export_metrics(root: Path, stage: Stage) -> None:
    """Construct CSV without the original append-only/misleading skip behavior."""
    paths = sorted((root / stage.name).glob("*/training_metadata.json"))
    if not paths:
        return
    items = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for obj in items:
        for key in ("feature_columns", "xgboost_params", "training_settings", "source_warnings"):
            obj[key] = json.dumps(obj[key], ensure_ascii=False, separators=(",", ":"))
    output = root / f"XGBoost_{stage.name}_daily_metrics.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(items[0].keys()))
        writer.writeheader()
        writer.writerows(items)


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    default_config = Path(__file__).resolve().parent / "configs" / "xgboost.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--stage", choices=[*STAGE_BY_NAME, "all"], default="all")
    parser.add_argument("--date", help="Single day in YYYY-MM-DD format")
    parser.add_argument("--start-date", default="1998-01-01")
    parser.add_argument("--end-date", default="1998-12-31")
    parser.add_argument("--output-dir", type=Path, default=default_config.parent.parent / "outputs" / "xgboost")
    parser.add_argument("--dry-run", action="store_true", help="Check input CSVs and features without fitting any model")
    parser.add_argument("--overwrite", action="store_true", help="Re-train existing date")
    parser.add_argument("--skip-missing", action="store_true", help="Explicitly skip dates without input CSVs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = cli(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.date and (args.start_date != "1998-01-01" or args.end_date != "1998-12-31"):
        raise SystemExit("Specify either --date or --start-date/--end-date")
    first = pd.Timestamp(args.date or args.start_date)
    last = pd.Timestamp(args.date or args.end_date)
    if last < first:
        raise SystemExit("End date precedes start date")
    if first.year < 1961 or last.year > 2019:
        LOG.warning("Requested range extends outside 1961–2019")
    config = args.config.resolve()
    input_dir, settings = read_config(config)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Prepared input directory does not exist: {input_dir}")
    root = args.output_dir.resolve()
    stages = STAGES if args.stage == "all" else (STAGE_BY_NAME[args.stage],)
    count, missing_count = 0, 0
    for stage in stages:
        for day in pd.date_range(first, last, freq="D"):
            csv_path = input_dir / f"{day:%Y%m%d}.csv"
            if not csv_path.is_file():
                if not args.skip_missing:
                    raise FileNotFoundError(f"Missing prepared station data: {csv_path}")
                LOG.warning("Skipping missing input: %s", csv_path)
                missing_count += 1
                continue
            if args.dry_run:
                x, _, n, warnings = read_day(csv_path, stage, settings)
                LOG.info("DRY RUN %s %s: rows=%s, features=%s, warnings=%s",
                         stage.name, day.date(), n, x.shape[1], warnings)
            else:
                train_day(stage, day, csv_path, root, settings, overwrite=args.overwrite)
            count += 1
        if not args.dry_run:
            export_metrics(root, stage)
    LOG.info("Finished %s %s jobs; skipped missing=%s", count,
             "validated" if args.dry_run else "training", missing_count)


if __name__ == "__main__":
    main()
