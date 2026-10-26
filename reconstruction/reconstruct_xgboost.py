#!/usr/bin/env python3
"""Generate XGBoost daily station estimates from saved Boosters and scalers."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reconstruction._daily_common import main_for_model


def predict_stage(stage: str, weights: Path, values: pd.DataFrame, scaler):
    """Load the date-specific XGBoost Booster without refitting."""
    from xgboost import Booster, DMatrix
    x = scaler.transform(values)
    model = Booster()
    model.load_model(str(weights))
    model.set_param({'nthread': 1})
    return np.asarray(model.predict(DMatrix(x, nthread=1))).reshape(-1)


if __name__ == '__main__':
    main_for_model('xgboost', predict_stage)
