"""Command-line interface: ``iot-city predict|evaluate|anomalies``."""

from __future__ import annotations

import argparse
import sys

from .data import extract_series, load_csv
from .model import AnomalyDetector, MovingAveragePredictor, mae, rmse


def _series(args) -> list[float]:
    records = load_csv(args.csv)
    return extract_series(records, args.sensor, args.metric)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="iot-city",
        description="Predictive IoT data analytics for smart city applications.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("predict", "Predict the next value for a sensor/metric"),
        ("evaluate", "Evaluate one-step-ahead predictions on a held-out tail"),
        ("anomalies", "List anomalies in a series"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--csv", required=True)
        p.add_argument("--sensor", required=True)
        p.add_argument("--metric", required=True)
        p.add_argument("--window", type=int, default=5)
        if name == "evaluate":
            p.add_argument("--test-size", type=int, default=6)
        if name == "anomalies":
            p.add_argument("--threshold", type=float, default=2.5)

    args = parser.parse_args(argv)
    try:
        series = _series(args)
        predictor = MovingAveragePredictor(window=args.window).fit(series)

        if args.command == "predict":
            print(f"next value: {predictor.predict_next():.3f}")
            return 0

        if args.command == "evaluate":
            if args.test_size < 1 or args.test_size >= len(series):
                raise ValueError("test-size must be between 1 and len(series) - 1")
            train, test = series[: -args.test_size], series[-args.test_size :]
            # One-step-ahead predictions, rolling the window forward.
            history = list(train)
            predictions: list[float] = []
            for _ in test:
                predictions.append(sum(history[-args.window :]) / args.window)
                history.append(test[len(predictions) - 1])
            print(f"MAE: {mae(test, predictions):.3f}")
            print(f"RMSE: {rmse(test, predictions):.3f}")
            return 0

        if args.command == "anomalies":
            predictions = predictor.predict_series(series)
            actual = series[args.window :]
            residuals = [a - p for a, p in zip(actual, predictions)]
            detector = AnomalyDetector(threshold=args.threshold).fit(residuals)
            hits = detector.find_anomalies(actual, predictions)
            if not hits:
                print("no anomalies found")
            for i in hits:
                print(f"anomaly at index {i + args.window}: value={actual[i]:.3f} predicted={predictions[i]:.3f}")
            return 0
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
