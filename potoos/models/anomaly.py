from typing import List
from dataclasses import dataclass
from luminol.modules.anomaly import Anomaly
from luminol.modules.time_series import TimeSeries


@dataclass
class DataPoint:
    """Class representing a time series data point."""

    timestamp: int  # Unix timestamp in milliseconds
    value: float


@dataclass
class TimeRange:
    """Class representing start, end timestamps and duration in ms"""
    start: int
    end: int
    duration: int


@dataclass
class MetaData:
    """Metadata about an anomaly detection run.

    Attributes:
        algorithm: The name of the algorithm used for anomaly detection
        data_points_analyzed: Number of data points that were analyzed
        anomalies_found: Number of anomalies detected in the data
        time_range_analyzed: TimeRange object containing start, end timestamps and duration in ms
    """
    algorithm: str
    data_points_analyzed: int
    anomalies_found: int
    time_range_analyzed: TimeRange


@dataclass
class AnomalyResult:
    """Class representing an anomaly detection result."""

    anomalies: List[Anomaly]
    scores: TimeSeries
    meta_data: MetaData
