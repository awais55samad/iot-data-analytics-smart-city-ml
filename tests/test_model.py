import unittest
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

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class TestDataLoading(unittest.TestCase):
    def test_load_csv_and_extract_series(self):
        records = load_csv(EXAMPLES / "sample_iot.csv")
        self.assertEqual(len(records), 24)
        series = extract_series(records, "temp-01", "temperature")
        self.assertEqual(len(series), 12)
        self.assertAlmostEqual(series[0], 24.0)

    def test_extract_unknown_series_raises(self):
        records = load_csv(EXAMPLES / "sample_iot.csv")
        with self.assertRaises(ValueError):
            extract_series(records, "no-such-sensor", "temperature")


class TestMovingAverage(unittest.TestCase):
    def test_predict_next_is_window_mean(self):
        predictor = MovingAveragePredictor(window=3).fit([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(predictor.predict_next(), 30.0)

    def test_predict_series(self):
        predictor = MovingAveragePredictor(window=2)
        predictions = predictor.predict_series([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(predictions, [1.5, 2.5])

    def test_too_short_series_raises(self):
        with self.assertRaises(ValueError):
            MovingAveragePredictor(window=5).fit([1.0, 2.0])


class TestLinearTrend(unittest.TestCase):
    def test_perfect_line_is_recovered(self):
        predictor = LinearTrendPredictor().fit([2.0, 4.0, 6.0, 8.0])
        self.assertAlmostEqual(predictor.slope, 2.0)
        self.assertAlmostEqual(predictor.intercept, 2.0)
        self.assertAlmostEqual(predictor.predict_next(), 10.0)

    def test_flat_series(self):
        predictor = LinearTrendPredictor().fit([5.0, 5.0, 5.0])
        self.assertAlmostEqual(predictor.predict_next(), 5.0)


class TestMetrics(unittest.TestCase):
    def test_mae_and_rmse(self):
        self.assertAlmostEqual(mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0)
        self.assertAlmostEqual(rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0)
        self.assertAlmostEqual(mae([0.0, 0.0], [1.0, 3.0]), 2.0)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            mae([1.0], [1.0, 2.0])


class TestAnomalyDetector(unittest.TestCase):
    def test_spike_is_anomaly(self):
        detector = AnomalyDetector(threshold=2.0).fit([0.1, -0.1, 0.0, 0.2, -0.2, 0.1])
        self.assertTrue(detector.is_anomaly(5.0))
        self.assertFalse(detector.is_anomaly(0.1))

    def test_find_anomalies_indices(self):
        detector = AnomalyDetector(threshold=2.0).fit([0.0, 0.1, -0.1, 0.0, 0.1, -0.1])
        hits = detector.find_anomalies(
            [10.0, 10.1, 20.0], [10.0, 10.0, 10.0]
        )
        self.assertEqual(hits, [2])


if __name__ == "__main__":
    unittest.main()
