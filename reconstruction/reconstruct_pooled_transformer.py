#!/usr/bin/env python3
"""Generate station estimates using pretrained pooled Transformer models.

Supports models trained over 1961–1990 or 1991–2019. Oracle uses observed
upstream variables and Operational uses preceding estimates. Historical mode
uses the original period-specific input columns. Pooled Tavg predictors are
alt, Lat, Lon, year, month and day. No model or scaler fitting is performed."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
STAGES=('Tavg','Tmax','Tmin','Ps','RH','SD')
TARGETS=dict(zip(STAGES,('MeanTEM','HighTEM','LowTEM','MeanPress','MeanRHU','Suntime')))
BASE=('alt','Lat','Lon','year','month','day')

def feature_columns(stage):
    return [*BASE,*[TARGETS[s] for s in STAGES[:STAGES.index(stage)]]]

def artifact_paths(root:Path,period:str,stage:str):
    t=TARGETS[stage]; p=period.replace('-','~');d=root/'pooled_transformer'/stage
    return (d/f'complete_transformer_model-{t}-{p}.h5',
            d/f'best_model_checkpoint-{t}-{p}.h5',
            d/f'{t}_scaler_{p}.save')

CONDITIONING = ('historical','oracle','operational')


def prepare_features(df:pd.DataFrame,period:str,stage:str,conditioning:str='historical'):
    """Return inputs in EXACT pooled training order (not general time-series tokens).

    Oracle and Operational differ only for stages after Tavg. The entry stage
    uses the SAME six static/calendar predictors for both conditioning modes.

    historical is deliberately period-specific to match as-supplied source:
    1961–1990 used estimated upstream inputs, while 1991–2019 used the
    upstream raw-named columns without testing their provenance.
    """
    if period not in ('1961-1990','1991-2019'):
        raise ValueError(f'Unsupported model period: {period}')
    if conditioning not in CONDITIONING:
        raise ValueError(f'Unsupported conditioning: {conditioning}')
    mode=conditioning
    if mode=='historical':
        mode='operational' if period=='1961-1990' else 'oracle'
    # This mapping for historical 1991–2019 means raw-named columns AS GIVEN,
    # not an assertion that they are proven independent station observations.
    upstream=STAGES[:STAGES.index(stage)]
    required=list(BASE)
    if conditioning=='historical' and period=='1961-1990':
        # The original scripts verified the raw-named columns were present
        # even though estimates replaced their values prior to inference.
        required += [TARGETS[s] for s in upstream]
    required += ([TARGETS[s] if mode=='oracle' else f'estimate_{TARGETS[s]}' for s in upstream])
    missing=set(required)-set(df.columns)
    if missing:
        raise ValueError(f'Missing required {conditioning} inputs for {stage}: {sorted(missing)}')
    x=df.loc[:,BASE].copy()
    for prior in upstream:
        predictor=TARGETS[prior]
        source=predictor if mode=='oracle' else f'estimate_{predictor}'
        x[predictor]=df[source]
    return x.loc[:,feature_columns(stage)]


def run(period:str,stage:str,input_dir:Path,output_dir:Path,artifact_root:Path,
        inspect_only=False,conditioning:str='historical'):
    if period not in ('1961-1990','1991-2019'):raise ValueError('Unsupported training period')
    if conditioning not in CONDITIONING:raise ValueError('Unsupported conditioning')
    files=sorted(input_dir.glob('*.csv'))
    if not files:raise FileNotFoundError(f'No CSV files in {input_dir}')
    full,checkpoint,scaler_file=artifact_paths(artifact_root,period,stage)
    for path in (full,checkpoint,scaler_file):
        if not path.is_file() and not inspect_only:raise FileNotFoundError(f'Missing previously trained model/scaler: {path}')
    results=[]
    model=scaler=None
    for path in files:
        df=pd.read_csv(path)
        x=prepare_features(df,period,stage,conditioning)
        bad=x.isna().any(axis=1)
        if bad.any():
            # Source scripts skipped an entire station file when any input cell was missing.
            results.append({'file':path.name,'status':'skipped_missing','rows':len(df)})
            continue
        if not inspect_only:
            if model is None:
                import tensorflow as tf
                model=tf.keras.models.load_model(str(full),compile=False)
                model.load_weights(str(checkpoint))
                scaler=joblib.load(scaler_file)
            xx=scaler.transform(x)
            pred=np.asarray(model.predict(xx,verbose=0)).reshape(-1)
            df[f'estimate_{TARGETS[stage]}']=pred
            output_dir.mkdir(parents=True,exist_ok=True)
            df.to_csv(output_dir/path.name,index=False)
        results.append({'file':path.name,'status':'inspected' if inspect_only else 'estimated','rows':len(df)})
    source = ('upstream_estimates' if conditioning=='operational' else
              'station_observations' if conditioning=='oracle' else
              'source_1961_1990_estimates' if period=='1961-1990' else 'source_1991_2019_columns_as_supplied')
    return {'stage':stage,'period':period,'conditioning':conditioning,
            'predictor_source':source,
            'entry_inputs':list(BASE) if stage=='Tavg' else None,
            'files':results,'artifacts':{'model':str(full),'weights':str(checkpoint),'scaler':str(scaler_file)}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--period',choices=['1961-1990','1991-2019'],required=True)
    p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--conditioning',choices=CONDITIONING,default='historical',
                   help='oracle=observed upstream; operational=estimated upstream; historical=period-specific original code')
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--artifacts-root',type=Path,default=ROOT/'model_artifacts')
    p.add_argument('--inspect-only',action='store_true')
    a=p.parse_args()
    print(json.dumps(run(a.period,a.stage,a.input_dir,a.output_dir,a.artifacts_root,a.inspect_only,
                         conditioning=a.conditioning),indent=2))
if __name__=='__main__':main()
