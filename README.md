# SMGNet-Code
SMGNet: A Mamba-Driven Graph Framework for Multi-Scale and Cross-Variable Long-Term Time Series Forecasting. The code will be made available in the near future.

## Overview
![Overview of GCTAM](fig_framework.png)

## Running Environment
* numpy
* matplotlib
* pandas
* scikit-learn
* torch==1.9.0

## Training
```bash
python train_GCTAM.py


## Dataset


| Dataset | Channels | Interval      | Timesteps         | Forecast Length |
|--------|-------------|----------|----------------|----------|
| ETTh1  |7| 1 hour |17420| 96,192,336,720|
| ETTh2  |7| 1 hour| 69680| 96,192,336,720|
| ETTm1  |7| 15 min| 17420| 96,192,336,720|
| ETTm2  |7| 15 min| 69680| 96,192,336,720|
| Weather  |21| 1 hour| 52696 | 96,192,336,720|
| Electricity  |321 |1 hour| 26304| 96,192,336,720|
| Traffic |862 | 1 hour |17544 |96,192,336,720|


