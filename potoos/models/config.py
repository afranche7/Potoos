from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class TimeSeriesConfig:
    """Configuration for time series data retrieval."""
    # Time range parameters
    start_time: Optional[Union[int, str]] = '-'  # Start time (Redis format or Unix ms)
    end_time: Optional[Union[int, str]] = '+'  # End time (Redis format or Unix ms)
    count: Optional[int] = 100  # Maximum number of points to retrieve

    # Aggregation parameters
    aggregation_type: Optional[str] = None  # 'avg', 'sum', 'min', 'max', etc.
    time_bucket: Optional[int] = None  # Time bucket size in milliseconds

    # Ordering parameters
    reverse_chronological: bool = False  # If True, newest points first


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    # Algorithm selection
    algorithm: str = 'derivative'  # Anomaly detection algorithm to use

    # Threshold settings
    threshold: float = 0.8  # Anomaly score threshold (0.0 to 1.0)
    max_score: Optional[float] = None  # Max score for normalization

    # For stream-based detection
    use_stream_detection: bool = False  # Whether to use streaming detection
    window_size: int = 100  # Window size for streaming detection
    step_size: int = 10  # Step size for sliding window

    # Multiple algorithm settings
    use_multiple_algorithms: bool = False  # Whether to use multiple algorithms
    algorithms: Optional[List[str]] = None  # List of algorithms to try