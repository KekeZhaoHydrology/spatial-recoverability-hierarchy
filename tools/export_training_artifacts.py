#!/usr/bin/env python3
"""Export completed training runs to the checkpoint names used for inference.

This tool copies saved XGBoost boosters and fitted scalers. For neural models
it serializes the already-trained BEST weights to a compatible HDF5 checkpoint
and verifies that the inference model can reload those weights. No fitting is
performed. Historical model files already present are not modified.
"""
from __future__ import annotations
import argparse
from datetime import date
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
STAGES = ('Tavg','Tmax','Tmin','Ps','RH','SD')
MODELS = ('mlp','xgboost','daily_transformer','pooled_transformer')
PATTERNS_PATH = ROOT/'reconstruction'/'model_filename_patterns.json'


def native_files(model: str, stage: str, stamp: str, root: Path):
    folder = root/stage/stamp
    if model == 'xgboost':
        from training.train_xgboost import STAGE_BY_NAME
        return folder/f'complete_model_{STAGE_BY_NAME[stage].target}_checkpoint-{stamp}.model', folder/f'{STAGE_BY_NAME[stage].target}_scaler_{stamp}.save'
    weights = 'best_weights.weights.h5' if model == 'mlp' else 'best_model.weights.h5'
    return folder/weights, folder/'feature_scaler.joblib'


def verify_weights(model, source_weights: Path, target_weights: Path, stage: str):
    if model=='mlp':
        from training.train_mlp import STAGE_BY_NAME, create_model
        builder=lambda: create_model(STAGE_BY_NAME[stage])
    else:
        from training.train_daily_transformer import STAGE_BY_NAME, create_model, Settings
        builder=lambda: create_model(STAGE_BY_NAME[stage], Settings())
    first=builder()
    first.load_weights(str(source_weights))
    # HDF5 export supplies the historical .h5 filename and best validation weights.
    # It is a full-model HDF5 container, not a bytewise copy of legacy checkpoint.
    first.save(str(target_weights))
    second=builder()
    second.load_weights(str(target_weights))
    for a,b in zip(first.get_weights(),second.get_weights()):
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-7)


def export_daily(model: str, stage: str, stamp: str, native_root: Path,
                 artifacts: Path, patterns: dict, *, overwrite=False, dry_run=False):
    if model not in MODELS[:3]:
        raise ValueError(f'Not a daily model: {model}')
    date.fromisoformat(f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}')
    if len(stamp)!=8 or not stamp.startswith('1998'):
        raise ValueError('This example artifact archive uses 1998 daily dates')
    source_model, source_scaler = native_files(model,stage,stamp,native_root)
    spec=patterns[model][stage]
    dest=artifacts/model/stage
    target_model=dest/spec['weights'].format(date=stamp)
    target_scaler=dest/spec['scaler'].format(date=stamp)
    output={'model':model,'stage':stage,'date':stamp,'source_model':str(source_model),
            'source_scaler':str(source_scaler),'output_model':str(target_model),
            'scaler':str(target_scaler),'status':'dry_run' if dry_run else 'exported'}
    if dry_run:
        return output
    if not source_model.is_file() or not source_scaler.is_file():
        raise FileNotFoundError(f'Missing trained model or fitted scaler: {source_model}, {source_scaler}')
    if (target_model.exists() or target_scaler.exists()) and not overwrite:
        raise FileExistsError(f'Output exists: {target_model} or {target_scaler}; use --overwrite only after backup')
    dest.mkdir(parents=True,exist_ok=True)
    if model=='xgboost':
        shutil.copy2(source_model,target_model)
    else:
        verify_weights(model,source_model,target_model,stage)
    shutil.copy2(source_scaler,target_scaler)
    return output


def export_pooled(stage: str,period: str,native_root: Path,artifacts: Path,
                  *,overwrite=False,dry_run=False):
    if period!='1991-2019':
        raise ValueError('Published example is 1991-2019')
    from training.train_pooled_transformer import STAGE_BY_NAME
    target=STAGE_BY_NAME[stage].target
    period_filename=period.replace('-','~')
    source=native_root/stage/period
    last=source/'last_epoch_model.keras'
    best=source/'best_weights.weights.h5'
    scaler=source/'feature_scaler.joblib'
    dest=artifacts/'pooled_transformer'/stage
    out_last=dest/f'complete_transformer_model-{target}-{period_filename}.h5'
    out_best=dest/f'best_model_checkpoint-{target}-{period_filename}.h5'
    out_scaler=dest/f'{target}_scaler_{period_filename}.save'
    details={'stage':stage,'period':period,'source_model':str(last),'source_weights':str(best),
             'source_scaler':str(scaler),'model':str(out_last),'weights':str(out_best),
             'scaler':str(out_scaler),'status':'dry_run' if dry_run else 'exported'}
    if dry_run:
        return details
    for src in (last,best,scaler):
        if not src.is_file():raise FileNotFoundError(src)
    if not overwrite and any(p.exists() for p in (out_last,out_best,out_scaler)):
        raise FileExistsError(f'Output exists in {dest}; back up before overwriting')
    import tensorflow as tf
    dest.mkdir(parents=True,exist_ok=True)
    model=tf.keras.models.load_model(str(last),compile=False)
    model.save(str(out_last))
    model.load_weights(str(best))
    model.save(str(out_best))
    verification=tf.keras.models.load_model(str(out_last),compile=False)
    verification.load_weights(str(out_best))
    for a,b in zip(model.get_weights(),verification.get_weights()):
        np.testing.assert_allclose(a,b,atol=1e-7,rtol=0)
    shutil.copy2(scaler,out_scaler)
    return details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--date',default='19980101',help='Daily models: YYYYMMDD')
    p.add_argument('--period',default='1991-2019',help='Pooled training period')
    p.add_argument('--native-root',type=Path,help='Directory containing output from training script')
    p.add_argument('--artifacts-root',type=Path,default=ROOT/'model_artifacts')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    native=args.native_root or ROOT/'training'/'outputs'/args.model
    if args.model=='pooled_transformer':
        result=export_pooled(args.stage,args.period,native,args.artifacts_root,
                             overwrite=args.overwrite,dry_run=args.dry_run)
    else:
        patterns=json.loads(PATTERNS_PATH.read_text(encoding='utf8'))
        result=export_daily(args.model,args.stage,args.date,native,args.artifacts_root,
                            patterns,overwrite=args.overwrite,dry_run=args.dry_run)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
