# IoT Data Analytics for Smart City Applications

Predictive, high-efficient IoT data analytics using machine learning
for smart city applications — time-series forecasting and anomaly
detection on IoT sensor data.

This is an **original, independent implementation inspired by** the 2025
paper *“A PREDICTIVE HIGH EFFICIENT IOT DATA ANALYTICS MODEL-USING
MACHINE LEARNING FOR SMART CITY APPLICATIONS”*. It is **not** the
official code from that paper, and it does not claim to reproduce its
results. It implements the core idea in a small, readable form:
IoT sensor readings are used to predict the next value and to flag
readings that deviate unexpectedly from the prediction.

## What it does

- Loads IoT sensor time-series from CSV
  (`timestamp,sensor_id,metric,value,unit`).
- Trains a lightweight predictor — a moving-average baseline and a
  simple linear-regression trend model — using only the Python
  standard library. No heavy dependencies, no network service.
- Predicts the next value for a sensor/metric series.
- Detects anomalies from prediction residuals (mean / standard
  deviation of residuals, configurable threshold).
- Evaluates predictions with MAE and RMSE.
- Ships with a command-line tool and tests.

## Quick start

```bash
python3 -m pip install -e .
python3 examples/demo.py
```

That loads `examples/sample_iot.csv`, trains on the first part of the
series, predicts the rest, and prints MAE / RMSE and any anomalies.

### Command line

```bash
# Predict the next value for one sensor/metric
iot-city predict --csv examples/sample_iot.csv --sensor temp-01 \
  --metric temperature --window 5

# Evaluate on a held-out tail of the series
iot-city evaluate --csv examples/sample_iot.csv --sensor temp-01 \
  --metric temperature --window 5 --test-size 6

# List anomalies in a series
iot-city anomalies --csv examples/sample_iot.csv --sensor temp-01 \
  --metric temperature --window 5 --threshold 2.5
```

## Project layout

```
src/iot_city/
  data.py   # TimeSeriesRecord, CSV loading, series extraction
  model.py  # MovingAveragePredictor, LinearTrendPredictor, AnomalyDetector, metrics
  cli.py    # Command-line interface
examples/
  sample_iot.csv
  demo.py
tests/
  test_model.py
```

## Tests

```bash
python3 -m pytest -q
# or, without pytest:
python3 -m unittest discover -s tests -v
```

## Author

**Engr. Muhammad Awais Khan** — Bahauddin Zakariya University, Multan (Pakistan)

Google Scholar profile: https://scholar.google.com/citations?hl=en&user=c4UawvgAAAAJ

## Status

Experimental research code, version 0.1.0. It is a teaching and
prototyping implementation of a small forecasting model, not a
production smart-city platform, and it has not been validated on
real city-scale data.

## License

MIT — see `LICENSE`.
