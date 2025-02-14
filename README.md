# SMGNet-Code
SMGNet: A Mamba-Enhanced Graph Approach for Multi-Variable Long-Term Time Series Forecasting
## Overview
![Overview of SMGNet](SMGNet.png)

## Running Environment
* numpy
* matplotlib
* pandas
* scikit-learn
* torch==2.4.1

## Training
python run_longExp.py

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

The datasets supporting this study are openly available.

- ETT: [ETT Data](https://github.com/zhouhaoyi/ETDataset)
- Traffic: [Traffic Data](http://pems.dot.ca.gov)
- Weather: [Weather Data](https://www.bgc-jena.mpg.de/wetter)
- Electricity: [Electricity Data](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)

