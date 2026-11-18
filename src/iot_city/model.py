"""Small forecasting models for IoT sensor time-series.

Standard library only. Two predictors are provided:

- :class:`MovingAveragePredictor`: predicts the mean of the last
  ``window`` values — a simple, robust baseline.
- :class:`LinearTrendPredictor`: fits a least-squares line
  ``value = slope * t + intercept`` to the series and extrapolates it.

:class:`AnomalyDetector` flags values whose prediction residual is
far from the residual mean, in units of residual standard deviation.
"""

from __future__ import annotations

import math
import statistics


def mae(actual: list[float], predicted: list[float]) -> float:
    """Mean absolute error."""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty lists of equal length")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: list[float], predicted: list[float]) -> float:
    """Root mean squared error."""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty lists of equal length")
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


class MovingAveragePredictor:
    """Predict the mean of the last ``window`` values."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self._history: list[float] = []

    def fit(self, series: list[float]) -> "MovingAveragePredictor":
        if len(series) < self.window:
            raise ValueError(f"need at least {self.window} values to fit")
        self._history = list(series)
        return self

    def predict_next(self) -> float:
        if len(self._history) < self.window:
            raise ValueError("predictor is not fitted")
        return sum(self._history[-self.window :]) / self.window

    def predict_series(self, series: list[float]) -> list[float]:
        """One-step-ahead predictions for every value after the window."""
        if len(series) <= self.window:
            raise ValueError("series is too short for one-step-ahead prediction")
        return [
            sum(series[i - self.window : i]) / self.window
            for i in range(self.window, len(series))
        ]


class LinearTrendPredictor:
    """Least-squares linear trend, extrapolated one step ahead."""

    def __init__(self) -> None:
        self.slope: float | None = None
        self.intercept: float | None = None
        self._n: int = 0

    def fit(self, series: list[float]) -> "LinearTrendPredictor":
        if len(series) < 2:
            raise ValueError("need at least 2 values to fit a trend")
        n = len(series)
        mean_t = (n - 1) / 2
        mean_y = sum(series) / n
        numerator = sum((t - mean_t) * (y - mean_y) for t, y in enumerate(series))
        denominator = sum((t - mean_t) ** 2 for t in range(n))
        self.slope = numerator / denominator if denominator else 0.0
        self.intercept = mean_y - self.slope * mean_t
        self._n = n
        return self

    def predict_next(self) -> float:
        if self.slope is None or self.intercept is None:
            raise ValueError("predictor is not fitted")
        return self.slope * self._n + self.intercept

    def predict_at(self, t: int) -> float:
        if self.slope is None or self.intercept is None:
            raise ValueError("predictor is not fitted")
        return self.slope * t + self.intercept


class AnomalyDetector:
    """Flag residuals that are ``threshold`` standard deviations away.

    Residuals are ``actual - predicted``. The detector is fitted on
    a list of residuals; a new residual is anomalous when its absolute
    z-score exceeds the threshold. A zero residual spread means only
    non-zero residuals are anomalous.
    """

    def __init__(self, threshold: float = 2.5) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self._mean: float = 0.0
        self._std: float = 0.0

    def fit(self, residuals: list[float]) -> "AnomalyDetector":
        if len(residuals) < 2:
            raise ValueError("need at least 2 residuals to fit")
        self._mean = statistics.fmean(residuals)
        self._std = statistics.pstdev(residuals)
        return self

    def is_anomaly(self, residual: float) -> bool:
        if self._std == 0.0:
            return residual != self._mean
        return abs(residual - self._mean) / self._std > self.threshold

    def find_anomalies(
        self, actual: list[float], predicted: list[float]
    ) -> list[int]:
        """Indices in ``actual`` whose residual is anomalous."""
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have equal length")
        return [
            i for i, (a, p) in enumerate(zip(actual, predicted)) if self.is_anomaly(a - p)
        ]
