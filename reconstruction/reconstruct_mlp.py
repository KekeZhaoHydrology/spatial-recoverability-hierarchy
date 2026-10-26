#!/usr/bin/env python3
"""Generate MLP daily station estimates using saved best weights and scalers."""
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
    """Create the original MLP architecture and load the date-specific weights."""
    from training.train_mlp import STAGE_BY_NAME, create_model
    x = scaler.transform(values)
    model = create_model(STAGE_BY_NAME[stage])
    model.load_weights(str(weights))
    return np.asarray(model.predict(x, verbose=0)).reshape(-1)


if __name__ == '__main__':
    main_for_model('mlp', predict_stage)
