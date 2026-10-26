#!/usr/bin/env python3
"""Train daily MLP models for the six meteorological targets.

Uses stage-specific prepared station inputs and saves dated model weights,
scalers and training summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("mlp_training")
SENTINEL = 32766


@dataclass(frozen=True)
class Stage:
    name: str
    target: str
    features: tuple[str, ...]
    hidden_units: tuple[int, ...]
    l2: float


STAGES: tuple[Stage, ...] = (
    Stage("Tavg", "MeanTEM", ("alt", "Lon", "Lat"), (128, 64, 32), 0.0),
    Stage("Tmax", "HighTEM", ("alt", "Lon", "Lat", "MeanTEM"), (128, 64, 32), 0.0),
    Stage("Tmin", "LowTEM", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM"), (128, 64, 32), 0.0),
    Stage("Ps", "MeanPress", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM", "LowTEM"), (128, 64, 32), 0.001),
    Stage("RH", "MeanRHU", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM", "LowTEM", "MeanPress"), (128, 64, 48, 32), 0.001),
    Stage("SD", "Suntime", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM", "LowTEM", "MeanPress", "MeanRHU"), (128, 64, 48, 32), 0.001),
)
STAGE_BY_NAME = {s.name: s for s in STAGES}


@dataclass(frozen=True)
class TrainingSettings:
    missing_value_indicator: int = SENTINEL
    validation_fraction: float = 0.1
    split_random_state: int = 42
    epochs: int = 600
    batch_size: int = 16
    patience: int = 80
    optimizer: str = "adam"
    loss: str = "mse"


def load_configuration(config_path: Path) -> tuple[dict[str, Path], TrainingSettings]:
    """Resolve all directories relative to config.json (not the shell cwd)."""
    with config_path.open("r", encoding="utf-8") as f:
        document = json.load(f)
    directory_map = document.get("stage_input_directories", {})
    names = set(STAGE_BY_NAME)
    missing = names - set(directory_map)
    unknown = set(directory_map) - names
    if missing or unknown:
        raise ValueError(f"Incorrect stage_input_directories; missing={sorted(missing)}, unknown={sorted(unknown)}")
    base = config_path.parent.resolve()
    dirs = {}
    for name, location in directory_map.items():
        value = Path(location).expanduser()
        dirs[name] = value.resolve() if value.is_absolute() else (base / value).resolve()
    settings = TrainingSettings(**document.get("training_settings", {}))
    if settings.validation_fraction <= 0 or settings.validation_fraction >= 1:
        raise ValueError("validation_fraction must lie between 0 and 1")
    if settings.optimizer != "adam" or settings.loss != "mse":
        raise ValueError("Original experiments use only optimizer='adam', loss='mse'.")
    return dirs, settings


def data_for_day(path: Path, stage: Stage, settings: TrainingSettings) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the exact ORIGINAL target-row selection and feature order.

    The original scripts fit StandardScaler on all target-observed rows BEFORE
    the 90/10 split. That is retained in train_for_day to preserve the recorded
    method, despite mild validation information leakage from the scaler.
    """
    frame = pd.read_csv(path)
    required = [*stage.features, stage.target]
    absent = [c for c in required if c not in frame.columns]
    if absent:
        raise ValueError(f"{path.name} lacks columns for {stage.name}: {absent}")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    non_missing = numeric.loc[numeric[stage.target] != settings.missing_value_indicator]
    if len(non_missing) < 12:
        raise ValueError(f"Insufficient target-observed station rows in {path}: {len(non_missing)}")
    if not np.isfinite(non_missing.to_numpy()).all():
        raise ValueError(f"Non-finite predictor/target values in {path}; check the original input CSV.")
    x = non_missing.loc[:, stage.features].to_numpy(dtype=np.float64, copy=True)
    y = non_missing.loc[:, stage.target].to_numpy(dtype=np.float64, copy=True)
    upstream_sentinel = (x == settings.missing_value_indicator).sum()
    if upstream_sentinel:
        LOG.warning("%s %s: %s predictor cells equal the missing-value sentinel; original code did not filter these.",
                    stage.name, path.name, upstream_sentinel)
    return x, y, len(frame)


def create_model(stage: Stage):
    """Keras architecture directly derived from the six original scripts."""
    import tensorflow as tf  # defer import so --dry-run works without TensorFlow

    layers: list[Any] = []
    for index, units in enumerate(stage.hidden_units):
        kwargs = {"activation": "relu"}
        if index == 0:
            kwargs["input_shape"] = (len(stage.features),)
        if stage.l2:
            kwargs["kernel_regularizer"] = tf.keras.regularizers.l2(stage.l2)
        layers.append(tf.keras.layers.Dense(units, **kwargs))
    layers.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="mse")
    return model


def _metrics(y_true: np.ndarray, y_predict: np.ndarray, prefix: str) -> dict[str, float]:
    yp = np.asarray(y_predict).reshape(-1)
    return {
        f"{prefix}_ME": float(np.mean(yp - y_true)),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, yp)),
        f"{prefix}_MSE": float(mean_squared_error(y_true, yp)),
        f"{prefix}_R2": float(r2_score(y_true, yp)),
    }


def artifacts_for_day(output_root: Path, stage: Stage, day_string: str) -> dict[str, Path]:
    folder = output_root / stage.name / day_string
    return {
        "directory": folder,
        "best_weights": folder / "best_weights.weights.h5",
        "last_epoch_full_model": folder / "last_epoch_model.h5",
        "best_full_model": folder / "best_model.h5",
        "scaler": folder / "feature_scaler.joblib",
        "metadata": folder / "training_metadata.json",
    }


def sha256_file(file_path: Path) -> str:
    """Fingerprint the stage-prepared daily input for reproducible skip logic."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_for_day(
    stage: Stage,
    single_date: pd.Timestamp,
    input_csv: Path,
    output_root: Path,
    settings: TrainingSettings,
    overwrite: bool = False,
    save_last_epoch_model: bool = False,
    save_best_full_model: bool = False,
) -> dict[str, Any]:
    """Train one daily stage, retaining original fitting and checkpoint semantics."""
    import tensorflow as tf

    stamp = single_date.strftime("%Y%m%d")
    paths = artifacts_for_day(output_root, stage, stamp)
    meta_path = paths["metadata"]
    if meta_path.exists() and not overwrite:
        required = ["best_weights", "scaler"]
        if save_last_epoch_model:
            required.append("last_epoch_full_model")
        if save_best_full_model:
            required.append("best_full_model")
        if all(paths[k].is_file() for k in required):
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            matches = (
                previous.get("training_settings") == asdict(settings)
                and previous.get("feature_columns") == list(stage.features)
                and previous.get("target_column") == stage.target
                and previous.get("architecture_hidden_units") == list(stage.hidden_units)
                and previous.get("kernel_l2_regularization") == stage.l2
                and previous.get("input_sha256") == sha256_file(input_csv)
            )
            if not matches:
                raise RuntimeError(f"Existing output for {stage.name} {stamp} was trained with different inputs/configuration; use --overwrite")
            LOG.info("Skip completed %s %s", stage.name, stamp)
            return previous
        raise RuntimeError(f"Incomplete output for {stage.name} {stamp}; inspect or use --overwrite")

    x, y, total_rows = data_for_day(input_csv, stage, settings)
    paths["directory"].mkdir(parents=True, exist_ok=True)

    # The scaler was originally fitted on all target-observed rows, before
    # train_test_split. Keep this to avoid silently changing experimental logic.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    joblib.dump(scaler, paths["scaler"])
    x_train, x_val, y_train, y_val = train_test_split(
        x_scaled, y,
        test_size=settings.validation_fraction,
        random_state=settings.split_random_state,
    )

    model = create_model(stage)
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(paths["best_weights"]),
        save_weights_only=True, monitor="val_loss", mode="min", save_best_only=True,
    )
    cb_early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=settings.patience, verbose=0,
    )
    history = model.fit(
        x_train, y_train, epochs=settings.epochs, batch_size=settings.batch_size,
        validation_data=(x_val, y_val), callbacks=[cb_checkpoint, cb_early_stop], verbose=0,
    )

    # Original order: final-epoch full-model save PRECEDES best-weight reload.
    # The optional last-epoch model reproduces that archival output exactly.
    # By default save only best weights + scaler to avoid redundant files.
    if save_last_epoch_model:
        model.save(str(paths["last_epoch_full_model"]))
    model.load_weights(str(paths["best_weights"]))
    if save_best_full_model:
        model.save(str(paths["best_full_model"]))

    train_predictions = model.predict(x_train, verbose=0)
    val_predictions = model.predict(x_val, verbose=0)
    vloss = list(history.history.get("val_loss", []))
    metrics: dict[str, Any] = {
        "date": stamp,
        "stage": stage.name,
        "target_column": stage.target,
        "feature_columns": list(stage.features),
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "all_target_observed_rows": int(len(y)),
        "source_csv_rows": int(total_rows),
        "input_filename": input_csv.name,
        "input_sha256": sha256_file(input_csv),
        "epochs_run": len(history.epoch),
        "best_epoch": (int(np.argmin(vloss)) + 1) if vloss else None,
        "best_validation_loss": float(np.min(vloss)) if vloss else None,
        "scaler_fitting_scope": "all_target_observed_rows_before_split (original implementation)",
        "training_settings": asdict(settings),
        "architecture_hidden_units": list(stage.hidden_units),
        "kernel_l2_regularization": stage.l2,
        "checkpoint_used_for_metrics": "best_weights.weights.h5",
        "saved_last_epoch_model": bool(save_last_epoch_model),
        "saved_best_full_model": bool(save_best_full_model),
        **_metrics(y_train, train_predictions, "train"),
        **_metrics(y_val, val_predictions, "validation"),
    }
    meta_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("Trained %s %s (n=%s, epochs=%s, best epoch=%s)",
             stage.name, stamp, len(y), metrics["epochs_run"], metrics["best_epoch"])
    return metrics


def export_metrics(output_root: Path, stage: Stage) -> None:
    """Rebuild one metrics CSV from per-day metadata (safe to rerun)."""
    meta_paths = sorted((output_root / stage.name).glob("*/training_metadata.json"))
    if not meta_paths:
        return
    records = []
    for path in meta_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        for key in ("feature_columns", "training_settings", "architecture_hidden_units"):
            record[key] = json.dumps(record[key], separators=(",", ":"))
        records.append(record)
    csv_file = output_root / f"MLP_{stage.name}_daily_metrics.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=root / "configs" / "mlp.json", help="Stage-specific input CSV directories")
    parser.add_argument("--stage", choices=[*STAGE_BY_NAME, "all"], default="all", help="One variable or all stages")
    parser.add_argument("--date", type=str, help="One day, e.g. 1998-01-01")
    parser.add_argument("--start-date", default="1998-01-01")
    parser.add_argument("--end-date", default="1998-12-31")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "mlp")
    parser.add_argument("--dry-run", action="store_true", help="Validate available input CSVs; do not train or import TensorFlow")
    parser.add_argument("--overwrite", action="store_true", help="Replace completed daily artifacts")
    parser.add_argument("--skip-missing", action="store_true", help="Explicitly skip dates without a CSV; logged and counted")
    parser.add_argument("--save-last-epoch-model", action="store_true", help="Also save a complete .h5 model at the FINAL epoch, as the legacy scripts did")
    parser.add_argument("--save-best-full-model", action="store_true", help="Also save a complete .h5 model after restoring BEST checkpoint weights")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.date and (args.start_date != "1998-01-01" or args.end_date != "1998-12-31"):
        raise SystemExit("Use either --date or --start-date/--end-date, not both")
    first = pd.Timestamp(args.date or args.start_date)
    last = pd.Timestamp(args.date or args.end_date)
    if last < first:
        raise SystemExit("End date is earlier than start date")
    if first.year < 1961 or last.year > 2019:
        LOG.warning("Selected dates extend beyond 1961–2019, the period of the original scripts")
    config_path = args.config.resolve()
    directories, settings = load_configuration(config_path)
    stages = STAGES if args.stage == "all" else (STAGE_BY_NAME[args.stage],)
    output_root = args.output_dir.resolve()
    dates = pd.date_range(first, last, freq="D")
    count = 0
    skipped_missing = 0

    for stage in stages:
        input_dir = directories[stage.name]
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory missing for {stage.name}: {input_dir}")
        for day in dates:
            filename = input_dir / f"{day:%Y%m%d}.csv"
            if not filename.is_file():
                if args.skip_missing:
                    LOG.warning("Missing input: %s", filename)
                    skipped_missing += 1
                    continue
                raise FileNotFoundError(f"Required stage input not found: {filename}")
            if args.dry_run:
                x, y, _ = data_for_day(filename, stage, settings)
                LOG.info("DRY RUN %s %s: %s rows, %s predictors", stage.name, day.date(), len(y), x.shape[1])
            else:
                train_for_day(stage, day, filename, output_root, settings,
                              overwrite=args.overwrite,
                              save_last_epoch_model=args.save_last_epoch_model,
                              save_best_full_model=args.save_best_full_model)
            count += 1
        if not args.dry_run:
            export_metrics(output_root, stage)
    LOG.info("Finished: %s daily stage(s) %s; missing skipped=%s", count,
             "validated" if args.dry_run else "processed", skipped_missing)


if __name__ == "__main__":
    main()
