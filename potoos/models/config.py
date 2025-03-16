from dataclasses import dataclass
from typing import Optional, Union, Dict, Literal
from luminol.modules.time_series import TimeSeries


@dataclass
class TimeSeriesConfig:
    """Configuration for RedisTimeSeries data retrieval.

    This class matches the parameters accepted by Redis TimeSeries' TS.RANGE and
    TS.REVRANGE commands for time series data retrieval.

    Attributes:
        from_time: Starting timestamp for range query. Can be Unix timestamp in ms
            or Redis special string format: '-' for earliest data point
        to_time: Ending timestamp for range query. Can be Unix timestamp in ms
            or Redis special string format: '+' for the current time
        count: Maximum number of data points to return
        aggregation_type: Type of aggregation to perform on the data. When specified,
            time_bucket must also be provided
        bucket_size_msec: Time bucket size for aggregation in milliseconds, required
            when aggregation is specified
        filter_by_ts: Optional list of specific timestamps to filter by
        filter_by_value: Optional range of values to filter by using min/max
        align: Controls timestamp alignment for aggregation buckets. When 'start',
            aligns with 'from_time'. When 'end', aligns with 'to_time'.
            When +/-int, aligns with specified Unix timestamp
        latest: When true, removes duplicates with the same timestamp, using only
            the latest value
        reversed: When true, returns time series data points in descending order (newest to oldest)
    """

    # Time range parameters
    from_time: Union[int, str] = '-'  # Start time (Unix ms timestamp or '-')
    to_time: Union[int, str] = '+'  # End time (Unix ms timestamp or '+')

    # Optional parameters
    count: Optional[int] = None  # Maximum number of samples to return

    # Aggregation parameters
    aggregation_type: Optional[Literal['avg', 'sum', 'min', 'max', 'range', 'count', 'first', 'last', 'std.p', 'std.s', 'var.p', 'var.s']] = None
    bucket_size_msec: Optional[int] = None  # Time bucket in milliseconds for aggregation

    # Filtering parameters
    filter_by_ts: Optional[list[int]] = None  # Specific timestamps to filter by
    filter_by_min_value: Optional[int] = None
    filter_by_max_value: Optional[int] = None

    # Additional parameters
    align: Optional[Union[Literal['start', 'end'], int]] = None  # Timestamp alignment
    latest: bool = False  # When True, returns only latest value for each timestamp
    bucket_timestamp: Optional[str] = None

    # Use TS.RANGE or TS.REVRANGE
    reversed: bool = False

    def __post_init__(self):
        """Validate the configuration parameters."""
        if self.aggregation_type is not None and self.bucket_size_msec is None:
            raise ValueError("time_bucket must be specified when aggregation is used")


@dataclass
class AnomalyDetectionConfig:
    """Configuration for luminol's AnomalyDetector.

    This data class encapsulates all configuration parameters needed for
    initializing a luminol.anomaly_detector.AnomalyDetector instance, except the TimeSeries data.

    Attributes:
        baseline_time_series: Optional baseline time series for comparison-based algorithms
        score_threshold: Threshold for anomaly scores (anomalies need higher scores than this)
        score_percentile_threshold: Percentile threshold for anomaly scores (alternative to absolute threshold)
        algorithm_name: Algorithm to use for anomaly detection
        algorithm_params: Additional parameters for the selected algorithm
        refine_algorithm_name: Algorithm to use for refining anomaly results
        refine_algorithm_params: Additional parameters for the refining algorithm
        algorithm_class: Custom algorithm class (alternative to using algorithm_name)
    """

    # Optional parameters with default values
    baseline_time_series: Optional[Union[Dict[int, float], TimeSeries]] = None
    score_only: bool = False
    score_threshold: Optional[float] = None
    score_percent_threshold: Optional[float] = None
    algorithm_name: Optional[str] = 'derivative_detector'
    algorithm_params: Optional[Dict] = None
    refine_algorithm_name: Optional[str] = None
    refine_algorithm_params: Optional[Dict] = None
    algorithm_class: Optional[object] = None

    def __post_init__(self):
        """Validate the configuration parameters.

        Ensures that at least one threshold method is specified and performs
        basic validation of the configuration parameters.
        """
        # Validate that the configuration makes sense
        if self.algorithm_name is None and self.algorithm_class is None:
            raise ValueError("Either algorithm_name or algorithm_class must be specified")
