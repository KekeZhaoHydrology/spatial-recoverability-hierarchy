#!/usr/bin/env python3
"""Shared station-file handling and predictor selection for daily models.

Oracle uses observed upstream variables; Operational uses preceding estimates.
The first stage (Tavg) uses alt, Lon and Lat only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = ('Tavg', 'Tmax', 'Tmin', 'Ps', 'RH', 'SD')
TARGETS = dict(zip(STAGES, ('MeanTEM','HighTEM','LowTEM','MeanPress','MeanRHU','Suntime')))
PREFIX = ('alt', 'Lon', 'Lat')
# Output field names follow the exact target variable used in TRAINING.
# These names do NOT modify the trained model's feature order or the historical
# checkpoint/scaler filenames; those are configured separately in JSON.
ESTIMATES = {
    family: {stage: f'estimate_{TARGETS[stage]}' for stage in STAGES}
    for family in ('mlp', 'xgboost', 'daily_transformer')
}

# The supplied ORIGINAL XGBoost estimate scripts wrote these legacy names.
# Input compatibility is deliberate; new outputs use the canonical names.
LEGACY_ESTIMATE_ALIASES = {
    'xgboost': {
        'estimate_HighTEM': ('estimate_MaxTEM',),
        'estimate_LowTEM': ('estimate_MinTEM-estimate',),
    },
}


def standardize_estimate_columns(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Normalize historical estimate column names WITHOUT changing values.

    The original training target names and the ordered feature names supplied
    to the saved StandardScaler remain untouched. If old and new column names
    coexist with different values, fail rather than silently selecting one.
    This operation is independent of Oracle/Operational conditioning: observed
    raw-named columns (MeanTEM, HighTEM, LowTEM, ...) are never changed here.
    """
    aliases = LEGACY_ESTIMATE_ALIASES.get(model, {})
    if not aliases:
        return frame
    frame = frame.copy()
    for canonical, old_names in aliases.items():
        for old in old_names:
            if old not in frame.columns:
                continue
            if canonical in frame.columns:
                current = frame[canonical]
                legacy = frame[old]
                # Concatenated station files can have a mix of old and new
                # schemas; fill missing cells only, never overwrite a value.
                overlap = current.notna() & legacy.notna()
                exact_match = current.eq(legacy)
                num_current = pd.to_numeric(current, errors='coerce')
                num_legacy = pd.to_numeric(legacy, errors='coerce')
                numeric_match = (
                    num_current.notna() & num_legacy.notna() &
                    num_current.eq(num_legacy)
                )
                if not bool((~overlap | exact_match | numeric_match).all()):
                    raise ValueError(
                        f'Conflicting estimate columns {canonical!r} and {old!r}; '
                        'cannot choose one without changing the historical values.'
                    )
                frame[canonical] = current.combine_first(legacy)
            else:
                frame[canonical] = frame[old]
            frame = frame.drop(columns=old)
    return frame



def feature_columns(stage: str) -> list[str]:
    return list(PREFIX) + [TARGETS[s] for s in STAGES[:STAGES.index(stage)]]


CONDITIONING = ('historical', 'oracle', 'operational')


def actual_input_columns(model: str, stage: str, conditioning: str = 'historical') -> list[str]:
    """Source columns used to construct a model input in its TRAINING feature order.

    Oracle: actual upstream station observations in their original feature columns.
    Operational: prior model estimates, never the withheld station observations.
    Historical: retains the author's daily scripts' estimated upstream values;
    historical XGBoost output-name aliases are translated to canonical names.
    Tavg has no upstream variables; thus the three modes give identical inputs.
    """
    if conditioning not in CONDITIONING:
        raise ValueError(f'Unrecognized conditioning: {conditioning!r}')
    if conditioning == 'oracle':
        return feature_columns(stage)
    return list(PREFIX) + [ESTIMATES[model][s] for s in STAGES[:STAGES.index(stage)]]


def read_station_files(folder: Path) -> pd.DataFrame:
    if not folder.is_dir():
        raise FileNotFoundError(f'Input folder absent: {folder}')
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.csv', '.xlsx'))
    if not files:
        raise FileNotFoundError(f'No station files in {folder}')
    frames = []
    for path in files:
        df = pd.read_csv(path) if path.suffix.lower()=='.csv' else pd.read_excel(path)
        for col in ('year','month','day'):
            if col not in df.columns:
                raise ValueError(f'{path.name}: no {col} column')
        df = df.dropna(subset=['year','month','day']).copy()
        if df.empty:continue
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['day'] = df['day'].astype(int)
        df['_source_file'] = path.name
        # The as-used sources constructed calendar keys from Y/M/D, not the input filename.
        df['_date_stamp'] = (df['year'].astype(str).str.zfill(4) +
                             df['month'].astype(str).str.zfill(2) +
                             df['day'].astype(str).str.zfill(2))
        frames.append(df)
    if not frames:raise ValueError(f'No dated station records in {folder}')
    return pd.concat(frames,ignore_index=True)


def resolve_artifacts(root: Path, model: str, stage: str, stamp: str, patterns: dict):
    base = root/model/stage
    spec = patterns[model][stage]
    return base / spec['weights'].format(date=stamp), base / spec['scaler'].format(date=stamp)


def model_predict(model_type: str, stage: str, weights: Path, values: pd.DataFrame, scaler):
    """Dispatch a model-specific predictor; each model has its own entry point."""
    from importlib import import_module
    backend = import_module(f'reconstruction.reconstruct_{model_type}')
    return backend.predict_stage(stage, weights, values, scaler)


def reconstruct(model: str, stage: str, input_dir: Path, output_dir: Path,
                artifact_root: Path, patterns: dict, first_date: str | None = None,
                last_date: str | None = None, *, inspect_only: bool = False,
                conditioning: str = 'historical', predictor=None) -> dict:
    if conditioning not in CONDITIONING:
        raise ValueError(f'Invalid conditioning: {conditioning!r}')
    df=read_station_files(input_dir)
    # Before selecting upstream predictors, translate old XGBoost result-column
    # aliases to TRAINING-target-aligned estimate_* names without changing data.
    df=standardize_estimate_columns(df,model)
    inputs = actual_input_columns(model,stage,conditioning)
    required=set(inputs)
    missing=required-set(df.columns)
    if missing:raise ValueError(f'{model}/{stage}: missing actual input columns: {sorted(missing)}')
    # Use preceding estimates only in the temporary input matrix. Keep the
    # original station-observed columns unchanged for every model family.
    ordered=feature_columns(stage)
    output_column = ESTIMATES[model][stage]
    # Preserve existing estimates outside the requested dates (and on rows
    # for which a new prediction cannot be generated).
    predictions = (pd.to_numeric(df[output_column], errors='coerce').copy()
                   if output_column in df.columns else
                   pd.Series(np.nan, index=df.index, dtype=float))
    used=0; skipped=0; lookup=[]
    for stamp,indexes in df.groupby('_date_stamp',sort=True).groups.items():
        if first_date and stamp<first_date:continue
        if last_date and stamp>last_date:continue
        weight,scaler_path=resolve_artifacts(artifact_root,model,stage,stamp,patterns)
        lookup.append((stamp,str(weight),str(scaler_path)))
        if not weight.is_file() or not scaler_path.is_file():
            if inspect_only:
                skipped+=1
                continue
            raise FileNotFoundError(f'Missing pretrained checkpoint/scaler for {model} {stage} {stamp}: {weight}; {scaler_path}')
        if inspect_only:
            used+=1
            continue
        x=df.loc[list(indexes),inputs].copy()
        x.columns=ordered # Rename only for the stored training Scaler feature names
        numeric=x.apply(pd.to_numeric,errors='coerce')
        mask=np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
        if not mask.any():
            skipped+=1;continue
        saved_scaler=joblib.load(scaler_path)
        pred=(predictor(stage,weight,numeric.loc[mask],saved_scaler) if predictor is not None else
             model_predict(model,stage,weight,numeric.loc[mask],saved_scaler))
        predictions.loc[numeric.index[mask]]=pred
        used+=1
    if not inspect_only:
        df[output_column]=predictions
        output_dir.mkdir(parents=True,exist_ok=True)
        for name,group in df.groupby('_source_file',sort=False):
            result=group.drop(columns=['_source_file','_date_stamp']).sort_values(['year','month','day'])
            path=output_dir/name
            if path.suffix.lower()=='.xlsx':result.to_excel(path,index=False)
            else:result.to_csv(path,index=False)
    return {'model':model,'stage':stage,'conditioning':conditioning,
            'predictor_source':('station_observations' if conditioning=='oracle' else 'upstream_estimates'),
            'entry_inputs':list(PREFIX) if stage=='Tavg' else None,
            'rows':len(df),'dates_with_artifacts':used,
            'dates_without_artifacts':skipped,'inspect_only':inspect_only,
            'example_lookup':lookup[:2]}


def main_for_model(model_type: str, predictor) -> None:
    """Run one estimate stage for a selected daily model family."""
    parser = argparse.ArgumentParser(
        description=f'Generate daily {model_type} station estimates from saved parameters.')
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--conditioning', choices=CONDITIONING, default='historical',
                        help='oracle: observed upstream; operational: previous estimates')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifacts-root', type=Path, default=ROOT/'model_artifacts')
    parser.add_argument('--first-date', default=None, help='YYYYMMDD')
    parser.add_argument('--last-date', default=None, help='YYYYMMDD')
    parser.add_argument('--inspect-only', action='store_true',
                        help='Validate checkpoint/scaler names and paths without loading them')
    args=parser.parse_args()
    patterns=json.loads((ROOT/'reconstruction/model_filename_patterns.json').read_text(encoding='utf8'))
    report=reconstruct(model_type,args.stage,args.input_dir,args.output_dir,
                       args.artifacts_root,patterns,args.first_date,args.last_date,
                       inspect_only=args.inspect_only,conditioning=args.conditioning,
                       predictor=predictor)
    print(json.dumps(report,indent=2))
