"""Test target names and compatibility with historical estimate fields."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reconstruction._daily_common import (
    STAGES, TARGETS, ESTIMATES, LEGACY_ESTIMATE_ALIASES, feature_columns,
    actual_input_columns, standardize_estimate_columns, reconstruct,
    resolve_artifacts,
)
from training import train_mlp, train_xgboost, train_daily_transformer

PATTERNS=json.loads((ROOT/'reconstruction/model_filename_patterns.json').read_text())

class FakeScaler:
    def transform(self, frame):
        return frame.to_numpy()

class EstimateNamingTests(unittest.TestCase):
    def test_all_18_daily_training_target_and_feature_orders_match(self):
        for model in (train_mlp,train_xgboost,train_daily_transformer):
            for stage in STAGES:
                self.assertEqual(model.STAGE_BY_NAME[stage].target,TARGETS[stage])
                self.assertEqual(list(model.STAGE_BY_NAME[stage].features), feature_columns(stage))

    def test_standardized_output_for_all_three_families(self):
        for family in ('mlp','xgboost','daily_transformer'):
            for stage in STAGES:
                self.assertEqual(ESTIMATES[family][stage],f'estimate_{TARGETS[stage]}')

    def test_train_checkpoint_scaler_paths_keep_original_names(self):
        for model in ('mlp','xgboost','daily_transformer'):
            for stage in STAGES:
                a,b=resolve_artifacts(ROOT/'model_artifacts',model,stage,'19980101',PATTERNS)
                self.assertEqual(PATTERNS[model][stage]['target'],TARGETS[stage])
                self.assertIn('{date}',PATTERNS[model][stage]['weights'])
                self.assertIn('{date}',PATTERNS[model][stage]['scaler'])
        a,b=resolve_artifacts(ROOT/'model_artifacts','xgboost','Tmin','19980101',PATTERNS)
        self.assertEqual(a.name,'complete_model_LowTEM_checkpoint-19980101.model')
        self.assertEqual(b.name,'MinTEM_scaler_19980101.save')

    def test_historical_xgboost_aliases_read_without_value_modification(self):
        df=pd.DataFrame({'estimate_MeanTEM':[9.], 'estimate_MaxTEM':[11.],
                         'estimate_MinTEM-estimate':[-3.],
                         'HighTEM':[1000.], 'LowTEM':[2000.]})
        out=standardize_estimate_columns(df,'xgboost')
        self.assertEqual(out['estimate_HighTEM'].tolist(),[11.])
        self.assertEqual(out['estimate_LowTEM'].tolist(),[-3.])
        self.assertNotIn('estimate_MaxTEM',out)
        self.assertNotIn('estimate_MinTEM-estimate',out)
        self.assertEqual(out['HighTEM'].tolist(),[1000.])
        self.assertEqual(out['LowTEM'].tolist(),[2000.])
        self.assertIn('estimate_MaxTEM',df.columns)  # no in-place mutation

    def test_mixed_old_and_new_input_files_are_coalesced_safely(self):
        df=pd.DataFrame({'estimate_HighTEM':[np.nan, 21.0, 30],
                         'estimate_MaxTEM':[19., np.nan,30.]})
        out=standardize_estimate_columns(df,'xgboost')
        self.assertEqual(out['estimate_HighTEM'].tolist(),[19.,21.,30.])
        self.assertNotIn('estimate_MaxTEM',out)

    def test_conflicting_old_and_new_values_stop(self):
        frame=pd.DataFrame({'estimate_HighTEM':[30.], 'estimate_MaxTEM':[20.]})
        with self.assertRaisesRegex(ValueError,'Conflicting estimate columns'):
            standardize_estimate_columns(frame,'xgboost')

    def test_aliases_do_not_change_non_xgboost_data(self):
        frame=pd.DataFrame({'estimate_MaxTEM':[1]})
        for model in ('mlp','daily_transformer'):
            out=standardize_estimate_columns(frame,model)
            self.assertEqual(list(out.columns),['estimate_MaxTEM'])

    def test_operational_columns_are_canonical_and_oracle_raw_unchanged(self):
        self.assertEqual(actual_input_columns('xgboost','Tmin','operational'),
                         ['alt','Lon','Lat','estimate_MeanTEM','estimate_HighTEM'])
        self.assertEqual(actual_input_columns('xgboost','Ps','historical'),
                         ['alt','Lon','Lat','estimate_MeanTEM','estimate_HighTEM','estimate_LowTEM'])
        self.assertEqual(actual_input_columns('xgboost','Tmin','oracle'),
                         ['alt','Lon','Lat','MeanTEM','HighTEM'])
        self.assertEqual(actual_input_columns('xgboost','Tavg','oracle'),
                         actual_input_columns('xgboost','Tavg','operational'))

    def test_real_runner_reads_legacy_prior_and_outputs_new_name(self):
        # Run a synthetic historical XGBoost Tmin step with fake saved artifacts.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp=root/'in'; inp.mkdir(); out=root/'out'
            pd.DataFrame({'year':[1998], 'month':[1], 'day':[1],
                'alt':[200.], 'Lon':[100.], 'Lat':[40.],
                'MeanTEM':[99.], 'HighTEM':[99.],
                'estimate_MeanTEM':[5.], 'estimate_MaxTEM':[12.]}
            ).to_csv(inp/'synthetic.csv',index=False)
            weight,scaler=resolve_artifacts(root/'artifacts','xgboost','Tmin','19980101',PATTERNS)
            weight.parent.mkdir(parents=True);weight.touch();scaler.touch()
            def fake_predict(family, stage, weights, inputs, scale):
                self.assertEqual(list(inputs.columns),feature_columns(stage))
                self.assertEqual(inputs.iloc[0]['MeanTEM'],5.)
                self.assertEqual(inputs.iloc[0]['HighTEM'],12.)
                return np.array([-2.5])
            with patch('reconstruction._daily_common.joblib.load',return_value=FakeScaler()), \
                 patch('reconstruction._daily_common.model_predict',side_effect=fake_predict):
                reconstruct('xgboost','Tmin',inp,out,root/'artifacts',PATTERNS,
                            conditioning='operational')
            output=pd.read_csv(out/'synthetic.csv')
            self.assertEqual(output.loc[0,'estimate_LowTEM'],-2.5)
            self.assertEqual(output.loc[0,'estimate_HighTEM'],12.)
            self.assertNotIn('estimate_MaxTEM',output.columns)
            self.assertNotIn('estimate_MinTEM-estimate',output.columns)
            self.assertEqual(output.loc[0,'HighTEM'],99.)  # untouched actual observation

    def test_legacy_low_tem_flows_into_pressure_with_unchanged_feature_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=root/'in';inp.mkdir();out=root/'out'
            pd.DataFrame({'year':[1998],'month':[1],'day':[1],
                'alt':[100], 'Lon':[110], 'Lat':[38],
                'MeanTEM':[99],'HighTEM':[99],'LowTEM':[99],
                'estimate_MeanTEM':[8],'estimate_MaxTEM':[15],
                'estimate_MinTEM-estimate':[2]}).to_csv(inp/'synthetic.csv',index=False)
            weight,scaler=resolve_artifacts(root/'artifacts','xgboost','Ps','19980101',PATTERNS)
            weight.parent.mkdir(parents=True);weight.touch();scaler.touch()
            def fake_predict(family, stage, weight, features, scaler):
                self.assertEqual(list(features.columns),feature_columns('Ps'))
                self.assertEqual(features.iloc[0][['MeanTEM','HighTEM','LowTEM']].tolist(),[8,15,2])
                return np.array([1012.])
            with patch('reconstruction._daily_common.joblib.load',return_value=FakeScaler()), \
                 patch('reconstruction._daily_common.model_predict',side_effect=fake_predict):
                reconstruct('xgboost','Ps',inp,out,root/'artifacts',PATTERNS,
                            conditioning='operational')
            frame=pd.read_csv(out/'synthetic.csv')
            self.assertEqual(frame.loc[0,'estimate_MeanPress'],1012.)
            self.assertEqual(frame.loc[0,'estimate_HighTEM'],15)
            self.assertEqual(frame.loc[0,'estimate_LowTEM'],2)
            self.assertNotIn('estimate_MinTEM-estimate',frame.columns)

if __name__=='__main__': unittest.main()
