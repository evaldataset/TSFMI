# Data Directory

This directory holds raw real-world time-series datasets used by the TSFMI
real-world stress tests. The datasets are **not redistributed** in this
repository (per their original licenses); users must obtain their own copy
from the canonical source.

| File | Source | License | Used by |
|---|---|---|---|
| `ETTh1.csv`, `ETTh2.csv` | https://github.com/zhouhaoyi/ETDataset | CC-BY 4.0 | `src/datasets/real_world.py` (trend, stationarity, seasonality, change-point) |
| `jena_climate_2009_2016.csv` | Max Planck Institute for Biogeochemistry | CC-BY 4.0 | weather real-world stress test |
| `LD2011_2014.txt` | UCI ML Repository (Electricity) | CC-BY 4.0 | electricity real-world stress test |
| `traffic.txt`, `traffic.txt.gz` | Caltrans PEMS | public domain | traffic real-world stress test |
| `exchange_rate.txt`, `exchange_rate.txt.gz` | per original distributors | research use | exchange-rate real-world stress test |
| `electricity_raw.zip` | UCI ML Repository | CC-BY 4.0 | upstream of LD2011_2014 |
| `ucr/` | UCR Time Series Archive | per archive terms | `scripts/run_ucr_baseline_validation.py` |

## How to populate

```bash
# ETTh1, ETTh2
wget -P data/ https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
wget -P data/ https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv

# Jena Climate
wget -P data/ https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
unzip -d data/ data/jena_climate_2009_2016.csv.zip

# Electricity (UCI)
wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
unzip -d data/ data/LD2011_2014.txt.zip

# UCR Archive (registration required)
# See https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ for access terms.
```

After populating, run `make smoke` to verify the loader paths.
