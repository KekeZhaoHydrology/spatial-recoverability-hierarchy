#!/usr/bin/env python3
"""Train daily Transformer models for the six meteorological targets.

Each station-day sample is represented as a single attention token.
Dated model weights, scalers and training summaries are saved."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("daily_transformer")
MISSING_SENTINEL = 32766


@dataclass(frozen=True)
class Stage:
    name: str
    target: str
    features: tuple[str, ...]
    dff: int


STAGES = (
    Stage("Tavg", "MeanTEM", ("alt", "Lon", "Lat"), 256),
    Stage("Tmax", "HighTEM", ("alt", "Lon", "Lat", "MeanTEM"), 280),
    Stage("Tmin", "LowTEM", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM"), 300),
    Stage("Ps", "MeanPress", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM", "LowTEM"), 300),
    Stage("RH", "MeanRHU", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM", "LowTEM", "MeanPress"), 300),
    Stage("SD", "Suntime", ("alt", "Lon", "Lat", "MeanTEM", "HighTEM", "LowTEM", "MeanPress", "MeanRHU"), 300),
)
STAGE_BY_NAME = {s.name: s for s in STAGES}


@dataclass(frozen=True)
class Settings:
    epochs: int = 500
    batch_size: int = 16
    patience: int = 80
    validation_fraction: float = 0.1
    split_random_state: int = 42
    num_heads: int = 3
    num_layers: int = 3
    dropout_rate: float = 0.1
    l2_regularization: float = 1e-6
    optimizer: str = "adam"
    loss: str = "mse"


def load_configuration(config_path: Path) -> tuple[Path, Settings]:
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if "input_directory" not in config:
        raise ValueError("config.json must contain 'input_directory'.")
    raw = Path(config["input_directory"]).expanduser()
    input_dir = raw.resolve() if raw.is_absolute() else (config_path.parent / raw).resolve()
    settings = Settings(**config.get("training_settings", {}))
    if settings.validation_fraction <= 0 or settings.validation_fraction >= 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if min(settings.epochs, settings.batch_size, settings.patience, settings.num_layers, settings.num_heads) <= 0:
        raise ValueError("Training settings that specify counts must be positive.")
    if settings.optimizer != "adam" or settings.loss != "mse":
        raise ValueError("The author-supplied experiments used Adam and MSE; changing these is unsupported here.")
    return input_dir, settings


def source_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def prepare_daily_data(csv_path: Path, stage: Stage, reject_sentinel: bool = False):
    """Retain original *all-row* X/y selection (no silent filtering)."""
    frame = pd.read_csv(csv_path)
    required = [*stage.features, stage.target]
    absent = [column for column in required if column not in frame.columns]
    if absent:
        raise ValueError(f"{csv_path.name}, {stage.name}: missing columns {absent}")
    values = frame.loc[:, required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{csv_path.name}, {stage.name}: NaN/inf found; check source preparation")
    sentinel_count = int((values.to_numpy() == MISSING_SENTINEL).sum())
    if sentinel_count:
        message = (f"{csv_path.name}, {stage.name}: {sentinel_count} cells equal "
                   f"{MISSING_SENTINEL}; original scripts did NOT mask missing rows")
        if reject_sentinel:
            raise ValueError(message)
        LOG.warning(message)
    if len(values) < 12:
        raise ValueError(f"{csv_path.name}, {stage.name}: too few rows ({len(values)})")
    x = values.loc[:, stage.features].to_numpy(dtype=np.float64, copy=True)
    y = values.loc[:, stage.target].to_numpy(dtype=np.float64, copy=True)
    return x, y, len(frame), sentinel_count


def create_model(stage: Stage, settings: Settings):
    """Reproduce EXACT logical layers from original daily Transformer scripts.

    Attention input is (batch, 1, number_of_features), not time-indexed tokens.
    In the originals 'd_model=3' was declared but NEVER used in create_transformer.
    """
    import tensorflow as tf  # lazy import allows structural tests without TF

    inputs = tf.keras.layers.Input(shape=(len(stage.features),))
    x = tf.keras.layers.Reshape((1, len(stage.features)))(inputs)
    for _ in range(settings.num_layers):
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=settings.num_heads,
            key_dim=len(stage.features),
            kernel_regularizer=tf.keras.regularizers.l2(settings.l2_regularization),
        )(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + x)
        x = tf.keras.layers.Dropout(settings.dropout_rate)(x)
        ffn = tf.keras.layers.Dense(
            stage.dff, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(settings.l2_regularization),
        )(x)
        ffn = tf.keras.layers.Dense(
            len(stage.features),
            kernel_regularizer=tf.keras.regularizers.l2(settings.l2_regularization),
        )(ffn)
        ffn = tf.keras.layers.Dropout(settings.dropout_rate)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn + x)
    x = tf.keras.layers.Reshape((len(stage.features),))(x)
    outputs = tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(settings.l2_regularization)
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def paths_for_day(root: Path, stage: Stage, stamp: str) -> dict[str, Path]:
    directory = root / stage.name / stamp
    return {
        "directory": directory,
        "scaler": directory / "feature_scaler.joblib",
        "weights": directory / "best_model.weights.h5",
        "last_model": directory / "last_epoch_model.h5",
        "best_model": directory / "best_model.h5",
        "metadata": directory / "training_metadata.json",
    }


def _metrics(y: np.ndarray, predicted: np.ndarray, label: str) -> dict[str, float]:
    yp = np.asarray(predicted).reshape(-1)
    return {
        f"{label}_ME": float(np.mean(yp - y)),
        f"{label}_MAE": float(mean_absolute_error(y, yp)),
        f"{label}_MSE": float(mean_squared_error(y, yp)),
        f"{label}_R2": float(r2_score(y, yp)),
    }


def train_one_day(
    stage: Stage,
    stamp: str,
    csv_path: Path,
    output_root: Path,
    settings: Settings,
    *,
    overwrite: bool = False,
    reject_sentinel: bool = False,
    save_last_epoch_model: bool = False,
    save_best_full_model: bool = False,
    training_seed: int | None = None,
) -> dict[str, Any]:
    import tensorflow as tf

    paths = paths_for_day(output_root, stage, stamp)
    file_sha = source_sha256(csv_path)
    original_setting = {
        "settings": asdict(settings),
        "target": stage.target,
        "features": list(stage.features),
        "dff": stage.dff,
        "input_sha256": file_sha,
        "training_seed": training_seed,
    }
    if paths["metadata"].exists() and not overwrite:
        previous = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        needed = ["weights", "scaler"]
        if save_last_epoch_model:
            needed.append("last_model")
        if save_best_full_model:
            needed.append("best_model")
        if all(paths[k].exists() for k in needed):
            if previous.get("run_signature") != original_setting:
                raise RuntimeError(f"{stage.name} {stamp}: existing output differs; use --overwrite")
            LOG.info("Skipping completed %s %s", stage.name, stamp)
            return previous
        raise RuntimeError(f"{stage.name} {stamp}: incomplete output; inspect or use --overwrite")

    x, y, n_rows, n_sentinel = prepare_daily_data(csv_path, stage, reject_sentinel)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    # Original scripts fitted the scaler BEFORE splitting, using all rows.
    x_scaled = scaler.fit_transform(x)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_scaled, y, test_size=settings.validation_fraction,
        random_state=settings.split_random_state,
    )
    joblib.dump(scaler, paths["scaler"])
    if training_seed is not None:
        tf.keras.utils.set_random_seed(training_seed)  # option only; historical code had no training seed
    model = create_model(stage, settings)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(paths["weights"]), save_weights_only=True,
            monitor="val_loss", mode="min", save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=settings.patience, verbose=0,
        ),
    ]
    history = model.fit(
        x_train, y_train,
        epochs=settings.epochs, batch_size=settings.batch_size,
        validation_data=(x_validation, y_validation),
        callbacks=callbacks, verbose=0,
    )

    # Same ordering as legacy files: last-epoch full-model saved BEFORE
    # reloading the best checkpoint, then evaluation used the BEST checkpoint.
    if save_last_epoch_model:
        model.save(str(paths["last_model"]))
    model.load_weights(str(paths["weights"]))
    if save_best_full_model:
        model.save(str(paths["best_model"]))

    predicted_train = model.predict(x_train, verbose=0)
    predicted_validation = model.predict(x_validation, verbose=0)
    losses = list(history.history.get("val_loss", []))
    metadata: dict[str, Any] = {
        "date": stamp,
        "stage": stage.name,
        "source_csv": csv_path.name,
        "run_signature": original_setting,
        "input_rows": n_rows,
        "sentinel_cells": n_sentinel,
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "scaler_fitting_scope": "all_input_rows_before_train_validation_split (original implementation)",
        "attention_sequence_length": 1,
        "attention_key_dim": len(stage.features),
        "epochs_run": len(history.epoch),
        "best_epoch": (int(np.argmin(losses)) + 1) if losses else None,
        "best_val_loss": float(np.min(losses)) if losses else None,
        "evaluation_weights": "best_model.weights.h5",
        "full_model_if_saved_before_reload": "last_epoch_model.h5",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        **_metrics(y_train, predicted_train, "train"),
        **_metrics(y_validation, predicted_validation, "validation"),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("Trained %s %s: n=%s, epochs=%s best=%s", stage.name, stamp, n_rows,
             metadata["epochs_run"], metadata["best_epoch"])
    return metadata


def export_metrics(output_root: Path, stage: Stage) -> None:
    records = []
    for path in sorted((output_root / stage.name).glob("*/training_metadata.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["run_signature"] = json.dumps(item["run_signature"], separators=(",", ":"))
        records.append(item)
    if not records:
        return
    path = output_root / f"Transformer_{stage.name}_daily_metrics.csv"
    fields = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=parent / "configs" / "daily_transformer.json")
    parser.add_argument("--stage", choices=[*STAGE_BY_NAME, "all"], default="all")
    parser.add_argument("--date", type=str, help="Train one day, e.g. 1998-01-01")
    parser.add_argument("--start-date", default="1998-01-01")
    parser.add_argument("--end-date", default="1998-12-31")
    parser.add_argument("--output-dir", type=Path, default=parent / "outputs" / "daily_transformer")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without importing TensorFlow")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-missing", action="store_true", help="Fail if an expected daily CSV is missing")
    parser.add_argument("--reject-sentinel", action="store_true", help="Fail on 32766 (changes original permissive data behavior)")
    parser.add_argument("--save-last-epoch-model", action="store_true", help="Save legacy-compatible pre-reload full model")
    parser.add_argument("--save-best-full-model", action="store_true", help="Save full model after loading best checkpoint")
    parser.add_argument("--training-seed", type=int, help="Optional TF seed; original neural training had no fixed seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.date and (args.start_date != "1998-01-01" or args.end_date != "1998-12-31"):
        raise SystemExit("Use --date OR --start-date/--end-date")
    start = pd.Timestamp(args.date or args.start_date)
    end = pd.Timestamp(args.date or args.end_date)
    if start > end:
        raise ValueError("start-date is later than end-date")
    input_dir, settings = load_configuration(args.config.resolve())
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Daily station CSV directory does not exist: {input_dir}")
    stages = STAGES if args.stage == "all" else (STAGE_BY_NAME[args.stage],)
    output_root = args.output_dir.resolve()
    processed, missing = 0, 0
    for stage in stages:
        for day in pd.date_range(start, end, freq="D"):
            stamp = f"{day:%Y%m%d}"
            input_csv = input_dir / f"{stamp}.csv"
            if not input_csv.exists():
                if args.strict_missing:
                    raise FileNotFoundError(input_csv)
                LOG.warning("Missing CSV; skipping %s", input_csv)
                missing += 1
                continue
            if args.dry_run:
                x, y, rows, missing_cells = prepare_daily_data(input_csv, stage, args.reject_sentinel)
                LOG.info("DRY RUN %s %s: %s rows, %s features, %s sentinel cells",
                         stage.name, stamp, rows, x.shape[1], missing_cells)
            else:
                train_one_day(
                    stage, stamp, input_csv, output_root, settings,
                    overwrite=args.overwrite, reject_sentinel=args.reject_sentinel,
                    save_last_epoch_model=args.save_last_epoch_model,
                    save_best_full_model=args.save_best_full_model,
                    training_seed=args.training_seed,
                )
            processed += 1
        if not args.dry_run:
            export_metrics(output_root, stage)
    LOG.info("Completed %d stage/day runs (%d missing daily inputs)", processed, missing)


if __name__ == "__main__":
    main()
