# A spatial recoverability hierarchy for multivariable meteorological forcing from sparse observations

Code accompanying the study of multivariable meteorological reconstruction over **1961–2019**. The repository provides training and station-level estimation programs for four model families: multilayer perceptron (MLP), XGBoost, daily Transformer, and pooled Transformer. Six meteorological variables are reconstructed in a sequential hierarchy using geographic information and previously recovered meteorological fields.

## Repository structure

```text
training/             Model architectures, training scripts, and configurations
reconstruction/       Model-specific estimation scripts and shared utilities
model_artifacts/      Pretrained model artifacts (download separately from GitHub Releases;
                      restore locally before pretrained-model estimation)
tests/                Tests for input features, naming, and estimation interfaces
tools/                Utility for exporting newly trained model artifacts
requirements.txt      Python dependencies
```

After downloading and extracting the release archives, `model_artifacts/` contains four model-family directories: `mlp/`, `xgboost/`, `daily_transformer/`, and `pooled_transformer/`. Each contains the six variable directories `Tavg`, `Tmax`, `Tmin`, `Ps`, `RH`, and `SD`. Original checkpoint and scaler filenames are preserved, and training performance metrics are provided as text files. The model artifacts are not stored in the source-code repository; they are distributed as ZIP assets in [Model artifacts v1.0.0](https://github.com/KekeZhaoHydrology/spatial-recoverability-hierarchy/releases/tag/v1.0.0).

## Variables and predictors

| Stage | Meteorological variable | Training target | Preceding meteorological predictors |
| --- | --- | --- | --- |
| 1 | Mean air temperature (Tavg) | `MeanTEM` | None |
| 2 | Maximum air temperature (Tmax) | `HighTEM` | `MeanTEM` |
| 3 | Minimum air temperature (Tmin) | `LowTEM` | `MeanTEM`, `HighTEM` |
| 4 | Atmospheric pressure (Ps) | `MeanPress` | `MeanTEM`, `HighTEM`, `LowTEM` |
| 5 | Relative humidity (RH) | `MeanRHU` | `MeanTEM`, `HighTEM`, `LowTEM`, `MeanPress` |
| 6 | Sunshine duration (SD) | `Suntime` | `MeanTEM`, `HighTEM`, `LowTEM`, `MeanPress`, `MeanRHU` |

For the MLP, XGBoost, and daily Transformer models, the initial Tavg predictors are `alt, Lon, Lat`. The pooled Transformer uses `alt, Lat, Lon, year, month, day`. At each subsequent stage, the preceding meteorological variables are appended in the order shown above. Input feature names and order must match those used to fit the saved scalers.

In **Oracle** estimation, preceding meteorological predictors are station observations (`MeanTEM`, `HighTEM`, etc.). In **Operational** estimation, they are estimates from earlier stages (`estimate_MeanTEM`, `estimate_HighTEM`, etc.). Both modes use the same trained model and scaler. Tavg has no upstream meteorological predictors and therefore uses identical inputs in both modes for a given model family.

Outputs follow the names `estimate_MeanTEM`, `estimate_HighTEM`, `estimate_LowTEM`, `estimate_MeanPress`, `estimate_MeanRHU`, and `estimate_Suntime`. Legacy XGBoost estimate-column names are also accepted as inputs. The optional `historical` mode follows the source-column conventions of the original estimation scripts.

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Training

Daily models are trained separately for each date and target variable; pooled Transformer models are fitted across a specified period, with one model per target variable. Training requires prepared station input files, configured in `training/configs/*.json`.

```bash
python training/train_mlp.py --stage Tavg --date YYYY-MM-DD
python training/train_xgboost.py --stage Tavg --date YYYY-MM-DD
python training/train_daily_transformer.py --stage Tavg --date YYYY-MM-DD
python training/train_pooled_transformer.py --stage Tavg
```

Replace `YYYY-MM-DD` with the date of the prepared training input. Training outputs are stored under `training/outputs/`. The optional `tools/export_training_artifacts.py` converts newly trained model files to the naming conventions used for estimation; the provided pretrained artifacts can be used directly.

## Station-level estimation

The estimation scripts load pretrained models and fitted scalers from `model_artifacts/<model>/<stage>/`. Daily model filenames are defined in [`model_filename_patterns.json`](reconstruction/model_filename_patterns.json); pooled Transformer filenames are defined in its estimation script.

MLP and daily Transformer estimation require the saved best checkpoint and scaler for each date. XGBoost requires the saved Booster and scaler. Pooled Transformer estimation requires the complete model, best checkpoint, and scaler for each variable. Model and scaler fitting are not performed during estimation.

Each command estimates one variable. For an Operational sequence, use the output of one stage as input to the next. For example, with MLP:

```bash
python reconstruction/reconstruct_mlp.py \
  --stage Tavg --conditioning operational \
  --input-dir /path/to/station_inputs --output-dir /path/to/mlp_tavg

python reconstruction/reconstruct_mlp.py \
  --stage Tmax --conditioning operational \
  --input-dir /path/to/mlp_tavg --output-dir /path/to/mlp_tmax
```

For Oracle Tmax estimation, supply the observed `MeanTEM` values instead:

```bash
python reconstruction/reconstruct_mlp.py \
  --stage Tmax --conditioning oracle \
  --input-dir /path/to/station_observations \
  --output-dir /path/to/mlp_tmax_oracle
```

The same stage and input/output options apply to `reconstruct_xgboost.py` and `reconstruct_daily_transformer.py`. Daily estimation accepts CSV or XLSX station files with `year`, `month`, and `day` fields. The pooled Transformer accepts CSV files and requires a model-period selection:

```bash
python reconstruction/reconstruct_pooled_transformer.py \
  --period 1991-2019 --stage Tavg --conditioning operational \
  --input-dir /path/to/station_inputs --output-dir /path/to/pooled_tavg
```

Station inputs must include the relevant geographic predictors and, for subsequent stages, the required preceding observations or estimates. Run estimation only for dates with available pretrained artifacts; the daily scripts also support `--first-date` and `--last-date`. Use separate input and output directories to preserve source records.

## Data and model availability

Station identifiers, observation periods, variable availability, and available metadata are reported in **Supplementary Tables 1–4** of the accompanying manuscript. Publicly available daily observations can be obtained from the original data providers. Third-party observation datasets and prepared station training/evaluation files are not redistributed in this repository.

Pretrained model parameters, fitted scalers, and training performance metrics are available separately under the repository's [Model artifacts v1.0.0](https://github.com/KekeZhaoHydrology/spatial-recoverability-hierarchy/releases/tag/v1.0.0), rather than in the source-code repository. The daily-model artifacts include 1998 as an example year, while the pooled Transformer artifacts correspond to the 1991–2019 training period.

### Restoring pretrained model artifacts

Download the five archives from the [Model artifacts v1.0.0 release](https://github.com/KekeZhaoHydrology/spatial-recoverability-hierarchy/releases/tag/v1.0.0) :

- `daily_transformer.zip` — daily Transformer artifacts
- `mlp.zip` — MLP artifacts
- `pooled_transformer.zip` — pooled Transformer artifacts
- `xgboost1.zip` — XGBoost artifacts for Tavg, Tmax, and Tmin
- `xgboost2.zip` — XGBoost artifacts for Ps, RH, and SD

Extract their contents under the repository root so that the final local structure is:

```text
model_artifacts/
├── daily_transformer/
├── mlp/
├── pooled_transformer/
└── xgboost/
```

Extract `xgboost1.zip` and `xgboost2.zip` into the same `model_artifacts/xgboost/` directory; retain the six stage subdirectories and any accompanying metric files. Depending on whether each archive already contains its model-family directory, choose the extraction destination accordingly to avoid nested duplicate directories (such as `model_artifacts/mlp/mlp/`). This local layout is required by the estimation scripts' default `--artifacts-root` setting. Training scripts can be run independently with prepared input data.
