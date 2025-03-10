import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from redis.client import Redis
from luminol.anomaly_detector import AnomalyDetector
from luminol.modules.time_series import TimeSeries


@dataclass
class DataPoint:
    """Class representing a time series data point."""
    timestamp: int # Unix timestamp in milliseconds
    value: float


@dataclass
class AnomalyResult:
    """Class representing an anomaly detection result."""
    timestamp: int  # Unix timestamp in milliseconds
    value: float | int
    anomaly_score: float
    is_anomaly: bool



class PotoosClient:
    def __init__(self, redis_client: Redis):
        self.redis_client: Redis = redis_client

    def get_last_n_points(self, key: bytes | str | memoryview, n: int) -> List[DataPoint]:
        """
        Get the last n points from a Redis time series.

        This is a simplified wrapper around get_latest_points() that focuses only
        on retrieving the most recent n data points, using default settings for
        time range and ordering.

        Args:
            key: The Redis time series key
            n: Number of most recent data points to return

        Returns:
            List of DataPoint objects containing timestamps and values,
            ordered chronologically (oldest to newest)
        """
        return self.get_latest_from_time_series(
            key=key,
            count=n,
            reverse_chronological=False
        )

    def get_latest_from_time_series(
            self,
            key: bytes | str | memoryview,
            count: int,
            start_time: int | str | None= None,
            end_time: int | str | None = None,
            aggregation_type: str | None = None,
            time_bucket: int | None = None,
            reverse_chronological: bool = False
    ) -> List[DataPoint]:
        """
        Fetch the latest N points from a Redis time series within a specified time range.

        Args:
            key: The Redis time series key
            count: Number of most recent data points to return
            start_time: Start time in Unix milliseconds or Redis time specifier ('-' for earliest)
                        If None, defaults to earliest point ('-')
            end_time: End time in Unix milliseconds or Redis time specifier ('+' for latest)
                      If None, defaults to current time in milliseconds
            aggregation_type: Type of aggregation ('avg', 'sum', 'min', 'max', etc.)
            time_bucket: Time bucket size in milliseconds for aggregation
            reverse_chronological: If True, return points newest to oldest
                                  If False, return points oldest to newest (chronological)

        Returns:
            List of DataPoint objects containing timestamps and values
        """
        # Set default time range if not specified
        if start_time is None:
            start_time = '-'  # Redis special value for earliest point

        if end_time is None:
            end_time = int(time.time() * 1000)  # Current time in milliseconds

        # Use revrange to get newest points first, for efficiency
        timestamps, values = self.redis_client.ts().revrange(
            key=key,
            from_time=start_time,
            to_time=end_time,
            count=count,
            aggregation_type=aggregation_type,
            bucket_size_msec=time_bucket
        )

        # Create DataPoint objects from the results
        data_points = [
            DataPoint(timestamp=ts, value=val)
            for ts, val in zip(timestamps, values)
        ]

        # If chronological order is requested, reverse the list
        if not reverse_chronological:
            data_points.reverse()

        return data_points

    def get_time_series(self, key: bytes | str | memoryview,
                        aggregation_type: str | None = None,
                        time_bucket: int | None = None,
                        start_time: int | str = '-',
                        end_time: int | str = '+',
                        count: int | None = 100) -> List[DataPoint]:
        """
        Fetch time series data from Redis with enhanced query options.

        Args:
            key: The Redis time series key
            aggregation_type: Type of aggregation ('avg', 'sum', 'min', 'max', etc.)
            time_bucket: Time bucket size in seconds for aggregation
            start_time: Start time in Unix milliseconds (None means '-' or earliest)
            end_time: End time in Unix milliseconds (None means '+' or latest)
            count: Maximum number of data points to return

        Returns:
            List of timestamps to values
        """
        timestamps, values = self.redis_client.ts().range(
            key=key, from_time=start_time, to_time=end_time,
            aggregation_type=aggregation_type,
            time_bucket=time_bucket,
            count=count
        )

        data_points: List[DataPoint] = []
        for timestamp, value in zip(timestamps, values):
            data_points.append(DataPoint(timestamp=timestamp, value=value))

        return data_points

    def detect_anomalies(
            self,
            data_points: List[DataPoint],
            threshold: float = 0.8,
            algorithm: str = 'derivative',
            max_score: float | None = None
    ) -> Tuple[List[AnomalyResult], Dict[str, Any]]:
        """
        Detect anomalies in a list of DataPoint objects using Luminol.

        Args:
            data_points: List of DataPoint objects containing timestamp and value
            threshold: Anomaly score threshold (0.0 to 1.0). Points with scores above
                       this threshold are considered anomalies
            algorithm: Anomaly detection algorithm to use. Options include:
                      'derivative' - Based on first and second derivatives (best for trends)
                      'exp_avg_detector' - Based on exponential moving averages
                      'bitmap_detector' - Pattern-based detector using SAX discretization
                      'default_detector' - Ensemble of the above methods
            max_score: Optional maximum score to normalize anomaly scores
                       If None, will use the maximum score detected

        Returns:
            Tuple containing:
              - List of AnomalyResult objects with anomaly scores and classification
              - Dictionary with metadata about the detection process
        """
        if len(data_points) < 4:
            raise ValueError("Not enough data points for anomaly detection (minimum 4 required)")

        data_dict = {point.timestamp / 1000: float(point.value) for point in data_points}

        time_series = TimeSeries(data_dict)

        detector = AnomalyDetector(time_series, algorithm_name=algorithm)

        anomalies = detector.get_anomalies()

        score_series = detector.get_all_scores()

        if max_score is None:
            max_score = max(score_series.values()) if score_series.values() else 1.0
            max_score = max(max_score, 0.001)

        results = []
        for point in data_points:
            point_sec = point.timestamp / 1000

            score = score_series.get(point_sec, 0)

            normalized_score = score / max_score

            is_anomaly = normalized_score >= threshold

            result = AnomalyResult(
                timestamp=point.timestamp,
                value=point.value,
                anomaly_score=normalized_score,
                is_anomaly=is_anomaly
            )

            results.append(result)

        anomaly_timestamps = []
        for anomaly in anomalies:
            start_time = int(anomaly.start_timestamp * 1000)
            end_time = int(anomaly.end_timestamp * 1000)
            anomaly_timestamps.append((start_time, end_time, anomaly.anomaly_score))

        metadata = {
            'algorithm': algorithm,
            'threshold': threshold,
            'data_points_analyzed': len(data_points),
            'anomalies_found': sum(1 for r in results if r.is_anomaly),
            'luminol_anomalies': len(anomalies),
            'anomaly_intervals': anomaly_timestamps,
            'max_score': max_score,
            'score_stats': {
                'min': min(r.anomaly_score for r in results) if results else 0,
                'max': max(r.anomaly_score for r in results) if results else 0,
                'avg': sum(r.anomaly_score for r in results) / len(results) if results else 0
            },
            'time_range': {
                'start': min(p.timestamp for p in data_points) if data_points else 0,
                'end': max(p.timestamp for p in data_points) if data_points else 0,
                'duration_ms': max(p.timestamp for p in data_points) - min(
                    p.timestamp for p in data_points) if data_points else 0
            }
        }

        return results, metadata

    def detect_anomalies_multiple_algorithms(
            self,
            data_points: List[DataPoint],
            threshold: float = 0.8,
            algorithms: List[str] | None = None
    ) -> Dict[str, Tuple[List[AnomalyResult], Dict]]:
        """
        Run anomaly detection with multiple algorithms and compare results.

        Args:
            data_points: List of DataPoint objects
            threshold: Anomaly score threshold
            algorithms: List of algorithm names to use
                       If None, uses all available algorithms

        Returns:
            Dictionary mapping algorithm names to their results and metadata
        """
        if algorithms is None:
            algorithms = ['derivative', 'exp_avg_detector', 'bitmap_detector', 'default_detector']

        results = {}
        for algorithm in algorithms:
            try:
                algorithm_results = self.detect_anomalies(
                    data_points=data_points,
                    threshold=threshold,
                    algorithm=algorithm
                )
                results[algorithm] = algorithm_results
            except Exception as e:
                print(f"Error with algorithm {algorithm}: {e}")

        return results

    def detect_and_analyze_stream(
            self,
            data_points: List[DataPoint],
            window_size: int = 100,
            threshold: float = 0.8,
            algorithm: str = 'derivative',
            step_size: int = 10
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in a sliding window over a stream of data points.
        This approach is better for analyzing long time series by examining
        local patterns.

        Args:
            data_points: List of DataPoint objects
            window_size: Number of points to include in each analysis window
            threshold: Anomaly score threshold
            algorithm: Detection algorithm to use
            step_size: Number of points to slide the window each time

        Returns:
            List of AnomalyResult objects for the entire dataset
        """
        if len(data_points) < window_size:
            results, _ = self.detect_anomalies(data_points, threshold, algorithm)
            return results

        all_results = []

        for i in range(0, len(data_points) - window_size + 1, step_size):
            window = data_points[i:i + window_size]
            window_results, _ = self.detect_anomalies(window, threshold, algorithm)

            if i + step_size < len(data_points) - window_size + 1:
                window_results = window_results[:step_size]

            all_results.extend(window_results)

        return all_results
