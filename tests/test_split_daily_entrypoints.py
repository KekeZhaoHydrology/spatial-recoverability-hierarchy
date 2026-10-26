"""Validate the three separate daily inference entry points."""
from __future__ import annotations
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reconstruction import _daily_common
from reconstruction import reconstruct_mlp, reconstruct_xgboost, reconstruct_daily_transformer


class IdentityScaler:
    def transform(self,values):
        return np.asarray(values)


class SplitDailyTests(unittest.TestCase):
    def test_entrypoint_help_has_no_model_option(self):
        for name in ('reconstruct_mlp.py','reconstruct_xgboost.py','reconstruct_daily_transformer.py'):
            p=subprocess.run([sys.executable,str(ROOT/'reconstruction'/name),'--help'],
                             capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn('--stage',p.stdout)
            self.assertNotIn('--model',p.stdout)

    def test_predictors_stay_in_separate_files(self):
        self.assertTrue(callable(reconstruct_mlp.predict_stage))
        self.assertTrue(callable(reconstruct_xgboost.predict_stage))
        self.assertTrue(callable(reconstruct_daily_transformer.predict_stage))

    def test_inspect_only_three_cli_scripts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp=root/'data'; inp.mkdir()
            pd.DataFrame({'year':[1998],'month':[1],'day':[1],
                          'alt':[100.], 'Lon':[110.], 'Lat':[36.]}).to_csv(inp/'synthetic.csv',index=False)
            for family,filename in [('mlp','reconstruct_mlp.py'),
                                    ('xgboost','reconstruct_xgboost.py'),
                                    ('daily_transformer','reconstruct_daily_transformer.py')]:
                cmd=[sys.executable,str(ROOT/'reconstruction'/filename),
                     '--stage','Tavg','--conditioning','oracle',
                     '--input-dir',str(inp),'--output-dir',str(root/'output'),
                     '--artifacts-root',str(root/'weights'),'--inspect-only']
                result=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                report=json.loads(result.stdout)
                self.assertEqual(report['model'],family)
                self.assertEqual(report['dates_without_artifacts'],1)
                self.assertIn(f'{family}/Tavg',report['example_lookup'][0][1].replace('\\','/'))

    def test_fixed_family_artifact_mapping(self):
        p=json.loads((ROOT/'reconstruction/model_filename_patterns.json').read_text())
        for family in ('mlp','xgboost','daily_transformer'):
            weight,scaler=_daily_common.resolve_artifacts(ROOT/'model_artifacts',family,'Tmin','19980101',p)
            self.assertEqual(weight.parent,ROOT/'model_artifacts'/family/'Tmin')
            self.assertTrue(scaler.name.endswith('19980101.save'))

    def test_mlp_scales_and_loads_best_weights(self):
        values=pd.DataFrame({'alt':[1.0], 'Lon':[100.0], 'Lat':[35.0]})
        class Dummy:
            def load_weights(self,p): self.path=p
            def predict(self,x,verbose=0): return np.array([[5.]])
        with patch('training.train_mlp.create_model',return_value=Dummy()):
            actual=reconstruct_mlp.predict_stage('Tavg',Path('weights.h5'),values,IdentityScaler())
        self.assertEqual(actual.tolist(),[5.])

    def test_backend_prediction_dispatch_is_correct(self):
        values=pd.DataFrame({'alt':[1.0], 'Lon':[100.0], 'Lat':[35.0]})
        for name,mod in [('mlp',reconstruct_mlp),('xgboost',reconstruct_xgboost),
                         ('daily_transformer',reconstruct_daily_transformer)]:
            with patch.object(mod,'predict_stage',return_value=np.array([2.0])) as predict:
                result=_daily_common.model_predict(name,'Tavg',Path('w'),values,IdentityScaler())
                self.assertEqual(result.tolist(),[2.0]);predict.assert_called_once()

if __name__=='__main__':unittest.main()
