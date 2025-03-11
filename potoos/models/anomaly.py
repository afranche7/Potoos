from dataclasses import dataclass


@dataclass
class DataPoint:
    """Class representing a time series data point."""
    timestamp: int  # Unix timestamp in milliseconds
    value: float


@dataclass
class AnomalyResult:
    """Class representing an anomaly detection result."""
    timestamp: int  # Unix timestamp in milliseconds
    value: float | int
    anomaly_score: float
    is_anomaly: bool
