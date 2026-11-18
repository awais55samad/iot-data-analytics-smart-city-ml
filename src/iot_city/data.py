"""IoT sensor time-series loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class TimeSeriesRecord:
    """One validated IoT sensor reading."""

    timestamp: str
    sensor_id: str
    metric: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        datetime.fromisoformat(self.timestamp)  # raises ValueError on bad input
        if not self.sensor_id.strip():
            raise ValueError("sensor_id must not be empty")
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if self.value != self.value:  # NaN check, without importing math
            raise ValueError("value must not be NaN")


def load_csv(path: str | Path) -> list[TimeSeriesRecord]:
    """Load records from a CSV file.

    Expected columns: ``timestamp,sensor_id,metric,value,unit``.
    """
    records: list[TimeSeriesRecord] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "sensor_id", "metric", "value", "unit"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must have columns: {sorted(required)}")
        for line_no, row in enumerate(reader, start=2):
            try:
                records.append(
                    TimeSeriesRecord(
                        timestamp=row["timestamp"].strip(),
                        sensor_id=row["sensor_id"].strip(),
                        metric=row["metric"].strip(),
                        value=float(row["value"]),
                        unit=row["unit"].strip(),
                    )
                )
            except (ValueError, KeyError) as exc:
                raise ValueError(f"CSV line {line_no}: {exc}") from exc
    if not records:
        raise ValueError("CSV contains no records")
    return records


def extract_series(
    records: list[TimeSeriesRecord], sensor_id: str, metric: str
) -> list[float]:
    """Return the values for one sensor/metric, in file order."""
    series = [
        r.value for r in records if r.sensor_id == sensor_id and r.metric == metric
    ]
    if not series:
        raise ValueError(f"no series found for sensor={sensor_id!r} metric={metric!r}")
    return series
