"""Demo: forecast temperature and flag anomalies from sample IoT data."""

from pathlib import Path

from iot_city import (
    AnomalyDetector,
    LinearTrendPredictor,
    MovingAveragePredictor,
    extract_series,
    load_csv,
    mae,
    rmse,
)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    records = load_csv(ROOT / "sample_iot.csv")
    series = extract_series(records, "temp-01", "temperature")

    train, test = series[:6], series[6:]
    predictor = MovingAveragePredictor(window=3).fit(train)
    print(f"moving-average next value: {predictor.predict_next():.2f} C")

    trend = LinearTrendPredictor().fit(train)
    print(f"linear-trend next value: {trend.predict_next():.2f} C")

    # One-step-ahead evaluation on the held-out tail.
    history = list(train)
    predictions = []
    for _ in test:
        predictions.append(sum(history[-3:]) / 3)
        history.append(test[len(predictions) - 1])
    print(f"MAE: {mae(test, predictions):.3f}  RMSE: {rmse(test, predictions):.3f}")

    # Anomaly detection on the full series.
    full_predictions = MovingAveragePredictor(window=3).predict_series(series)
    actual = series[3:]
    residuals = [a - p for a, p in zip(actual, full_predictions)]
    detector = AnomalyDetector(threshold=2.5).fit(residuals)
    hits = detector.find_anomalies(actual, full_predictions)
    print(f"anomalies: {hits if hits else 'none'}")


if __name__ == "__main__":
    main()
