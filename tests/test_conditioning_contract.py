"""Test Oracle and Operational predictor selection with synthetic values."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reconstruction._daily_common import reconstruct, resolve_artifacts, feature_columns, actual_input_columns, STAGES
from reconstruction.reconstruct_pooled_transformer import prepare_features, BASE as POOLED_BASE

class IdentityScaler:
    def transform(self, x):
        return x.to_numpy()

class ConditioningTests(unittest.TestCase):
    def test_tavg_entry_has_no_upstream_meteorology_all_daily_models(self):
        for family in ('mlp', 'xgboost', 'daily_transformer'):
            for conditioning in ('historical', 'oracle', 'operational'):
                self.assertEqual(actual_input_columns(family, 'Tavg', conditioning),
                                 ['alt', 'Lon', 'Lat'])

    def test_tavg_entry_pooled_has_calendar_inputs_in_both_modes(self):
        row = pd.DataFrame({'alt':[300], 'Lat':[40], 'Lon':[110],
                            'year':[1998], 'month':[1], 'day':[10],
                            'MeanTEM':[8], 'estimate_MeanTEM':[99]})
        for period in ('1961-1990', '1991-2019'):
            out = [prepare_features(row, period, 'Tavg', mode)
                   for mode in ('oracle', 'operational', 'historical')]
            self.assertEqual(list(out[0].columns),list(POOLED_BASE))
            pd.testing.assert_frame_equal(out[0],out[1])
            pd.testing.assert_frame_equal(out[1],out[2])

    def test_oracle_is_observed_and_operational_is_estimated_daily(self):
        patterns=json.loads((ROOT/'reconstruction/model_filename_patterns.json').read_text())
        for family in ('mlp', 'xgboost', 'daily_transformer'):
            for mode,want in (('oracle',11),('operational',21)):
                with self.subTest(family=family,mode=mode):
                    with tempfile.TemporaryDirectory() as tmp:
                        t=Path(tmp); src=t/'input';src.mkdir()
                        pd.DataFrame({'year':[1998], 'month':[1], 'day':[1],
                                      'alt':[30], 'Lon':[100], 'Lat':[40],
                                      'MeanTEM':[10], 'estimate_MeanTEM':[20]}).to_csv(src/'synthetic.csv', index=False)
                        artifacts=t/'artifacts'
                        a,b=resolve_artifacts(artifacts,family,'Tmax','19980101',patterns)
                        a.parent.mkdir(parents=True); a.touch(); b.touch()
                        with patch('reconstruction._daily_common.joblib.load',return_value=IdentityScaler()), \
                             patch('reconstruction._daily_common.model_predict',
                                   side_effect=lambda model,stage,weights,x,scaler: x['MeanTEM'].to_numpy()+1):
                            result=reconstruct(family,'Tmax',src,t/'out',artifacts,patterns,
                                               conditioning=mode)
                        generated=pd.read_csv(t/'out/synthetic.csv')
                        estimate={'mlp':'estimate_HighTEM','xgboost':'estimate_HighTEM',
                                  'daily_transformer':'estimate_HighTEM'}[family]
                        self.assertEqual(generated.loc[0,estimate],want)
                        self.assertEqual(generated.loc[0,'MeanTEM'], 10)
                        self.assertEqual(result['conditioning'],mode)

    def test_pooled_upstream_oracle_operational_and_historical(self):
        df=pd.DataFrame({'alt':[100], 'Lat':[30], 'Lon':[110],
                         'year':[1998], 'month':[8], 'day':[5],
                         'MeanTEM':[10], 'estimate_MeanTEM':[20],
                         'HighTEM':[15], 'estimate_HighTEM':[25]})
        for period in ('1961-1990','1991-2019'):
            self.assertEqual(prepare_features(df,period,'Tmin','oracle').iloc[0]['MeanTEM'],10)
            self.assertEqual(prepare_features(df,period,'Tmin','oracle').iloc[0]['HighTEM'],15)
            self.assertEqual(prepare_features(df,period,'Tmin','operational').iloc[0]['MeanTEM'],20)
            self.assertEqual(prepare_features(df,period,'Tmin','operational').iloc[0]['HighTEM'],25)
        self.assertEqual(prepare_features(df,'1961-1990','Tmin','historical').iloc[0]['MeanTEM'],20)
        self.assertEqual(prepare_features(df,'1991-2019','Tmin','historical').iloc[0]['MeanTEM'],10)

    def test_oracle_does_not_require_estimate_column(self):
        row=pd.DataFrame({'alt':[100], 'Lon':[110], 'Lat':[30],
                          'year':[1998], 'month':[1], 'day':[1], 'MeanTEM':[12]})
        out=prepare_features(row,'1991-2019','Tmax','oracle')
        self.assertEqual(out.iloc[0]['MeanTEM'],12)
        with self.assertRaisesRegex(ValueError,'estimate_MeanTEM'):
            prepare_features(row,'1991-2019','Tmax','operational')

    def test_invalid_conditioning_rejected(self):
        with self.assertRaises(ValueError):
            actual_input_columns('mlp','Tavg','nonsense')

    def test_partial_date_keeps_previous_estimates(self):
        """Restricting a run must not erase estimates for dates not rerun."""
        patterns = json.loads((ROOT/'reconstruction/model_filename_patterns.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root/'input'
            inp.mkdir()
            data = pd.DataFrame({
                'year': [1998, 1998], 'month': [1, 1], 'day': [1, 2],
                'alt': [100, 100], 'Lon': [110, 110], 'Lat': [35, 35],
                'MeanTEM': [8., 9.], 'estimate_MeanTEM': [80., 90.],
            })
            data.to_csv(inp/'station.csv', index=False)
            weights, scaler = resolve_artifacts(root/'models', 'mlp', 'Tavg', '19980101', patterns)
            weights.parent.mkdir(parents=True)
            weights.touch()
            scaler.touch()
            with patch('reconstruction._daily_common.joblib.load', return_value=IdentityScaler()), \
                 patch('reconstruction._daily_common.model_predict', return_value=[11.]):
                reconstruct('mlp', 'Tavg', inp, root/'out', root/'models', patterns,
                            '19980101', '19980101', conditioning='operational')
            result = pd.read_csv(root/'out/station.csv')
            self.assertEqual(result['estimate_MeanTEM'].tolist(), [11., 90.])
            self.assertEqual(result['MeanTEM'].tolist(), [8., 9.])

    def test_daily_transformer_keeps_raw_station_columns(self):
        """Operational input estimation must not replace observed inputs on disk."""
        patterns = json.loads((ROOT/'reconstruction/model_filename_patterns.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root/'input'
            inp.mkdir()
            pd.DataFrame({
                'year': [1998], 'month': [1], 'day': [1],
                'alt': [100.], 'Lon': [110.], 'Lat': [35.],
                'MeanTEM': [8.], 'estimate_MeanTEM': [18.],
            }).to_csv(inp/'station.csv', index=False)
            weights, scaler = resolve_artifacts(root/'models', 'daily_transformer', 'Tmax', '19980101', patterns)
            weights.parent.mkdir(parents=True)
            weights.touch()
            scaler.touch()
            def predict(model, stage, weight, inputs, normalizer):
                self.assertEqual(inputs['MeanTEM'].tolist(), [18.])
                return [19.]
            with patch('reconstruction._daily_common.joblib.load', return_value=IdentityScaler()), \
                 patch('reconstruction._daily_common.model_predict', side_effect=predict):
                reconstruct('daily_transformer', 'Tmax', inp, root/'out', root/'models', patterns,
                            conditioning='operational')
            result = pd.read_csv(root/'out/station.csv')
            self.assertEqual(result['MeanTEM'].tolist(), [8.])
            self.assertEqual(result['estimate_MeanTEM'].tolist(), [18.])
            self.assertEqual(result['estimate_HighTEM'].tolist(), [19.])

    def test_pooled_training_and_inference_stage_features_match(self):
        from training import train_pooled_transformer
        from reconstruction.reconstruct_pooled_transformer import feature_columns as pooled_columns
        for stage in STAGES:
            self.assertEqual(list(train_pooled_transformer.STAGE_BY_NAME[stage].features), pooled_columns(stage))

    def test_pooled_artifact_names_and_scales(self):
        from reconstruction.reconstruct_pooled_transformer import artifact_paths
        for stage, target in [('Tavg','MeanTEM'), ('Tmax','HighTEM'),
                              ('Tmin','LowTEM'), ('Ps','MeanPress'),
                              ('RH','MeanRHU'), ('SD','Suntime')]:
            model, weights, scaler = artifact_paths(ROOT/'model_artifacts', '1991-2019', stage)
            self.assertEqual(model.parent, ROOT/'model_artifacts'/'pooled_transformer'/stage)
            self.assertEqual(model.name, f'complete_transformer_model-{target}-1991~2019.h5')
            self.assertEqual(weights.name, f'best_model_checkpoint-{target}-1991~2019.h5')
            self.assertEqual(scaler.name, f'{target}_scaler_1991~2019.save')

if __name__=='__main__':
    unittest.main()
