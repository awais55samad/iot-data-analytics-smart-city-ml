"""Predictive IoT data analytics for smart city applications."""

from .data import TimeSeriesRecord, extract_series, load_csv
from .model import AnomalyDetector, LinearTrendPredictor, MovingAveragePredictor, mae, rmse

__all__ = [
    "AnomalyDetector",
    "LinearTrendPredictor",
    "MovingAveragePredictor",
    "TimeSeriesRecord",
    "extract_series",
    "load_csv",
    "mae",
    "rmse",
]
__version__ = "0.1.0"
