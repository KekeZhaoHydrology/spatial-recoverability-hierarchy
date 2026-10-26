#!/usr/bin/env python3
"""Train pooled Transformer models for six meteorological targets.

One model is trained per target using pooled station-day records.
Predictors include geographic, calendar and preceding meteorological fields."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("pooled_transformer_training")
SENTINEL = 32766  # Known missing-value convention in companion data-processing scripts.
BASE_FEATURES = ("alt", "Lat", "Lon", "year", "month", "day")


@dataclass(frozen=True)
class Stage:
    name: str
    target: str
    features: tuple[str, ...]
    epochs: int
    d_model_legacy: int  # Original scripts declared this, but active model did NOT use it.


STAGES: tuple[Stage, ...] = (
    Stage("Tavg", "MeanTEM", BASE_FEATURES, 30, 12),
    Stage("Tmax", "HighTEM", BASE_FEATURES + ("MeanTEM",), 100, 12),
    Stage("Tmin", "LowTEM", BASE_FEATURES + ("MeanTEM", "HighTEM"), 100, 12),
    Stage("Ps", "MeanPress", BASE_FEATURES + ("MeanTEM", "HighTEM", "LowTEM"), 100, 12),
    Stage("RH", "MeanRHU", BASE_FEATURES + ("MeanTEM", "HighTEM", "LowTEM", "MeanPress"), 100, 15),
    Stage("SD", "Suntime", BASE_FEATURES + ("MeanTEM", "HighTEM", "LowTEM", "MeanPress", "MeanRHU"), 100, 15),
)
STAGE_BY_NAME = {stage.name: stage for stage in STAGES}


@dataclass(frozen=True)
class Settings:
    validation_fraction: float = 0.1
    split_random_state: int = 42
    batch_size: int = 12000
    num_heads: int = 3
    num_layers: int = 3
    dff: int = 300
    optimizer: str = "adam"
    loss: str = "mse"


def load_config(path: Path) -> tuple[Path, Path, int, int, Settings]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    base = path.resolve().parent
    def locate(key: str) -> Path:
        value = Path(config[key]).expanduser()
        return (value if value.is_absolute() else base / value).resolve()
    first, last = int(config.get("start_year", 1961)), int(config.get("end_year", 1990))
    if first > last:
        raise ValueError("start_year must not exceed end_year")
    settings = Settings(**config.get("training_settings", {}))
    if not (0 < settings.validation_fraction < 1):
        raise ValueError("validation_fraction must be in (0, 1)")
    if settings.num_heads != 3 or settings.num_layers != 3 or settings.dff != 300:
        raise ValueError("Original pooled Transformer uses 3 heads, 3 layers and dff=300")
    if settings.batch_size != 12000 or settings.optimizer != "adam" or settings.loss != "mse":
        raise ValueError("Default reproduction requires batch_size=12000, adam, and mse")
    return locate("input_directory"), locate("output_directory"), first, last, settings


def input_files(directory: Path, first_year: int, last_year: int, order: str = "filesystem") -> list[Path]:
    """Find every .csv whose first four filename characters are a valid year.

    'filesystem' reflects original os.listdir iteration. --file-order sorted is
    recommended for future reproducible reruns, but may change the historical
    row order and thereby change train_test_split with a fixed random_state.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Input folder not found: {directory}")
    names = os.listdir(directory)
    if order == "sorted":
        names.sort()
    elif order != "filesystem":
        raise ValueError(f"Unknown file order: {order}")
    found: list[Path] = []
    for name in names:
        if not name.endswith(".csv"):
            continue
        try:
            year = int(name[:4])
        except ValueError:
            LOG.warning("Skipping CSV without four-digit year prefix: %s", name)
            continue
        path = directory / name
        if first_year <= year <= last_year and path.is_file():
            found.append(path)
    if not found:
        raise FileNotFoundError(f"No matching CSV files for {first_year}-{last_year} in {directory}")
    return found


def file_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    return [{"file": p.name, "size_bytes": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in paths]


def fingerprint(manifest: list[dict[str, Any]]) -> str:
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_pooled_data(paths: list[Path], *, cache_path: Path | None = None) -> pd.DataFrame:
    """Concatenate station-day rows as in the original implementation.

    A versioned cache is optional and validated by file metadata. The original
    scripts used a shared, unvalidated combined_data_1961~1990.pkl.
    """
    manifest = file_manifest(paths)
    token = fingerprint(manifest)
    if cache_path and cache_path.is_file():
        meta = cache_path.with_suffix(".manifest.json")
        if meta.is_file():
            cached = json.loads(meta.read_text(encoding="utf-8"))
            if cached.get("manifest_sha256") == token:
                LOG.info("Using validated cache: %s", cache_path)
                return pd.read_pickle(cache_path)
        LOG.warning("Ignoring stale/unverified cache: %s", cache_path)
    frames = []
    for number, path in enumerate(paths, 1):
        LOG.info("Reading %s/%s: %s", number, len(paths), path.name)
        frames.append(pd.read_csv(path))
    pooled = pd.concat(frames, ignore_index=True)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pooled.to_pickle(cache_path)
        cache_path.with_suffix(".manifest.json").write_text(
            json.dumps({"manifest_sha256": token, "files": manifest}, indent=2), encoding="utf-8")
    return pooled


def training_arrays(frame: pd.DataFrame, stage: Stage) -> tuple[np.ndarray, np.ndarray]:
    required = [*stage.features, stage.target]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for {stage.name}: {missing}")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{stage.name} input contains missing/non-numeric/infinite values; no silent imputation is applied")
    sentinel_count = int((numeric == SENTINEL).sum().sum())
    if sentinel_count:
        LOG.warning("%s includes %d values equal to %s; original script did not filter them. "
                    "Check upstream preparation and station exclusion rules.",
                    stage.name, sentinel_count, SENTINEL)
    x = numeric[list(stage.features)].to_numpy(dtype=np.float64, copy=True)
    y = numeric[stage.target].to_numpy(dtype=np.float64, copy=True)
    if len(y) < 12:
        raise ValueError(f"Only {len(y)} rows for {stage.name}; insufficient for a 10%% validation split")
    return x, y


def build_model(n_features: int, settings: Settings):
    """Exact ACTIVE single-token, three-block model from the original scripts.

    Legacy PositionalEncoding(sequence_length=10_000_000) was instantiated on
    an unrelated, unconnected tensor. It never entered the trained model, so it
    is excluded rather than consuming hundreds of MB in a dead graph branch.
    """
    import tensorflow as tf  # optional during validation/dry run
    inputs = tf.keras.Input(shape=(n_features,))
    x = tf.keras.layers.Reshape((1, n_features))(inputs)
    for _ in range(settings.num_layers):
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=settings.num_heads, key_dim=n_features)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + x)
        ff = tf.keras.layers.Dense(settings.dff, activation="relu")(x)
        ff = tf.keras.layers.Dense(n_features)(ff)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff + x)
    x = tf.keras.layers.Reshape((n_features,))(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=settings.optimizer, loss=settings.loss)
    return model


def metrics(y_true: np.ndarray, pred: np.ndarray, prefix: str) -> dict[str, float]:
    predicted = np.asarray(pred).reshape(-1)
    return {
        f"{prefix}_ME": float(np.mean(predicted - y_true)),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, predicted)),
        f"{prefix}_MSE": float(mean_squared_error(y_true, predicted)),
        f"{prefix}_R2": float(r2_score(y_true, predicted)),
    }


def train_stage(frame: pd.DataFrame, stage: Stage, settings: Settings,
                output_root: Path, first_year: int, last_year: int,
                manifest_sha: str, *, overwrite: bool = False,
                save_best_full_model: bool = False) -> dict[str, Any]:
    import tensorflow as tf
    destination = output_root / stage.name / f"{first_year}-{last_year}"
    destination.mkdir(parents=True, exist_ok=True)
    last_model = destination / "last_epoch_model.keras"
    best_weights = destination / "best_weights.weights.h5"
    best_model = destination / "best_model.keras"
    scaler_file = destination / "feature_scaler.joblib"
    metadata_file = destination / "training_metadata.json"
    if any(p.exists() for p in (last_model, best_weights, best_model, scaler_file, metadata_file)) and not overwrite:
        raise FileExistsError(f"Output already exists in {destination}; use --overwrite after verifying provenance")
    x, y = training_arrays(frame, stage)

    # Preserve original behavior: scaler fitted to all rows BEFORE 90/10 split.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    x_train, x_val, y_train, y_val = train_test_split(
        scaled, y, test_size=settings.validation_fraction,
        random_state=settings.split_random_state)
    joblib.dump(scaler, scaler_file)

    model = build_model(x.shape[1], settings)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_weights), save_weights_only=True,
        monitor="val_loss", mode="min", save_best_only=True)
    history = model.fit(x_train, y_train, epochs=stage.epochs,
                        batch_size=settings.batch_size,
                        validation_data=(x_val, y_val), callbacks=[ckpt], verbose=2)

    # IMPORTANT: original code saved final-epoch *full model* before restoring
    # best validation weights; evaluation was performed after restoration.
    model.save(str(last_model))
    model.load_weights(str(best_weights))
    if save_best_full_model:
        model.save(str(best_model))
    predicted_train = model.predict(x_train, verbose=0)
    predicted_val = model.predict(x_val, verbose=0)
    val_losses = history.history.get("val_loss", [])
    record: dict[str, Any] = {
        "stage": stage.name, "target_column": stage.target,
        "feature_columns": list(stage.features),
        "training_period": [first_year, last_year],
        "combined_rows": int(len(y)),
        "training_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
        "training_settings": asdict(settings),
        "requested_epochs": stage.epochs, "epochs_run": len(history.epoch),
        "legacy_declared_d_model_not_used": stage.d_model_legacy,
        "active_token_sequence_length": 1,
        "legacy_positional_encoding_not_connected": True,
        "input_manifest_sha256": manifest_sha,
        "scaler_fitting_scope": "all_pooled_rows_before_train_test_split (original behavior)",
        "model_save_semantics": "last_epoch_model.keras saved BEFORE best_weights.weights.h5 reload",
        "best_epoch": int(np.argmin(val_losses)) + 1 if val_losses else None,
        "best_val_loss": float(np.min(val_losses)) if val_losses else None,
        "tensorflow_version": tf.__version__,
        **metrics(y_train, predicted_train, "train"),
        **metrics(y_val, predicted_val, "validation"),
    }
    metadata_file.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def save_results_csv(output_root: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with (output_root / "pooled_training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            prepared = dict(record)
            for field in ("feature_columns", "training_period", "training_settings"):
                prepared[field] = json.dumps(prepared[field], separators=(",", ":"))
            writer.writerow(prepared)


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().parent / "configs" / "pooled_transformer.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--stage", choices=["all", *STAGE_BY_NAME], default="all")
    parser.add_argument("--start-year", type=int, help="Override year range for a separate run")
    parser.add_argument("--end-year", type=int, help="Override year range for a separate run")
    parser.add_argument("--file-order", choices=["filesystem", "sorted"], default="filesystem",
                        help="filesystem follows legacy os.listdir; sorted is reproducible but may change split")
    parser.add_argument("--dry-run", action="store_true", help="Inspect source CSVs/feature integrity; do not import TensorFlow")
    parser.add_argument("--cache", action="store_true", help="Use validated pooled pickle cache to avoid reloading CSVs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-best-full-model", action="store_true")
    parser.add_argument("--tf-seed", type=int,
                        help="Optional new-run seed; ORIGINAL scripts had no TensorFlow seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    input_dir, output_dir, first_year, last_year, settings = load_config(args.config.resolve())
    first_year = args.start_year if args.start_year is not None else first_year
    last_year = args.end_year if args.end_year is not None else last_year
    if first_year > last_year:
        raise ValueError("start-year cannot exceed end-year")
    if (first_year, last_year) != (1961, 1990):
        LOG.warning("Using %d-%d rather than the ORIGINAL pooled experiment's 1961-1990", first_year, last_year)
    paths = input_files(input_dir, first_year, last_year, args.file_order)
    manifest = file_manifest(paths)
    manifest_sha = fingerprint(manifest)
    LOG.info("Pooled training period: %d-%d; %d daily CSV files; order=%s",
             first_year, last_year, len(paths), args.file_order)
    stages = STAGES if args.stage == "all" else (STAGE_BY_NAME[args.stage],)
    cache_path = (output_dir / "cache" / f"pooled_{first_year}-{last_year}_{args.file_order}.pkl") if args.cache else None
    pooled = load_pooled_data(paths, cache_path=cache_path)
    LOG.info("Combined %d rows", len(pooled))
    for stage in stages:
        x, y = training_arrays(pooled, stage)
        LOG.info("Validated %s: target=%s, rows=%s, predictors=%s", stage.name, stage.target, len(y), x.shape[1])
    if args.dry_run:
        LOG.info("DRY RUN passed; no models trained and no TensorFlow imported")
        return

    if args.tf_seed is not None:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(args.tf_seed)
        LOG.warning("A new TensorFlow random seed was set; historical training did not set one")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_manifest.json").write_text(
        json.dumps({"start_year": first_year, "end_year": last_year,
                    "file_order": args.file_order, "manifest_sha256": manifest_sha,
                    "source_files": manifest}, indent=2) + "\n", encoding="utf-8")
    records = []
    for stage in stages:
        LOG.info("Starting pooled model %s (%d epochs)", stage.name, stage.epochs)
        records.append(train_stage(pooled, stage, settings, output_dir,
                                   first_year, last_year, manifest_sha,
                                   overwrite=args.overwrite,
                                   save_best_full_model=args.save_best_full_model))
    save_results_csv(output_dir, records)
    LOG.info("Finished %s pooled model(s)", len(records))


if __name__ == "__main__":
    main()
