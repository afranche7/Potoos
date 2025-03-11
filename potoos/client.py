import time
from redis.client import Redis
from luminol.anomaly_detector import AnomalyDetector
from luminol.modules.time_series import TimeSeries
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from .models.config import TimeSeriesConfig, AnomalyDetectionConfig
from .models.anomaly import DataPoint, AnomalyResult


class PotoosClient:
    def __init__(
            self,
            redis_client: Redis,
            default_ts_config: Optional[TimeSeriesConfig] = None,
            default_anomaly_config: Optional[AnomalyDetectionConfig] = None
    ):
        self.redis_client: Redis = redis_client
        self.default_ts_config = default_ts_config or TimeSeriesConfig()
        self.default_anomaly_config = default_anomaly_config or AnomalyDetectionConfig()
        self.check_time_series_module()

    def check_time_series_module(self):
        """
        Verifies that the RedisTimeSeries module is loaded on the Redis server.
        Raises an exception if the module is not available.
        """
        modules = self.redis_client.module_list()

        has_time_series = any(m.get('name') == 'timeseries' for m in modules)

        if not has_time_series:
            raise ModuleNotFoundError(
                "RedisTimeSeries module is not loaded on the Redis server. "
                "Please load the module before using time series functionality."
            )

    def monitor(
            self,
            key: bytes | str | memoryview,
            ts_config: Optional[TimeSeriesConfig] = None,
            anomaly_config: Optional[AnomalyDetectionConfig] = None,
            callback: Optional[Callable[[List[AnomalyResult], Dict[str, Any]], None]] = None
    ) -> Tuple[List[AnomalyResult], Dict[str, Any]]:
        """
        Monitors a time series key by fetching data and performing anomaly detection.

        Args:
            key: The Redis time series key to monitor
            ts_config: Configuration for time series data retrieval
            anomaly_config: Configuration for anomaly detection
            callback: Optional callback function that receives anomaly results

        Returns:
            Tuple containing:
                - List of AnomalyResult objects
                - Dictionary with metadata about the detection process
        """
        ts_config = ts_config or self.default_ts_config
        anomaly_config = anomaly_config or self.default_anomaly_config

        data_points: List[DataPoint] = self.get_time_series(key, config=ts_config)

        if not data_points:
            return [], {'error': 'No data points retrieved'}

        results = None
        metadata = None

        if anomaly_config.use_multiple_algorithms:
            algorithm_results = self.detect_anomalies_multiple_algorithms(
                data_points=data_points,
                config=anomaly_config
            )

            primary_algorithm = anomaly_config.algorithm
            if primary_algorithm in algorithm_results:
                results, metadata = algorithm_results[primary_algorithm]
            else:
                first_algo = next(iter(algorithm_results))
                results, metadata = algorithm_results[first_algo]

            if metadata:
                metadata['multi_algorithm_results'] = algorithm_results

        elif anomaly_config.use_stream_detection:
            results = self.detect_and_analyze_stream(
                data_points=data_points,
                config=anomaly_config
            )
            metadata = {
                'detection_method': 'stream',
                'window_size': anomaly_config.window_size,
                'step_size': anomaly_config.step_size,
                'algorithm': anomaly_config.algorithm,
                'threshold': anomaly_config.threshold,
                'data_points': len(data_points)
            }
        else:
            results, metadata = self.detect_anomalies(
                data_points=data_points,
                config=anomaly_config
            )

        if callback and results is not None and metadata is not None:
            callback(results, metadata)

        return results, metadata

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
            count: Optional[int] = None,
            start_time: Optional[Union[int, str]] = None,
            end_time: Optional[Union[int, str]] = None,
            aggregation_type: Optional[str] = None,
            time_bucket: Optional[int] = None,
            reverse_chronological: Optional[bool] = None,
            config: Optional[TimeSeriesConfig] = None
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
            config: TimeSeriesConfig object to use. If provided, its values override individual params.

        Returns:
            List of DataPoint objects containing timestamps and values
        """
        ts_config = config or self.default_ts_config

        actual_count = count if count is not None else ts_config.count
        actual_start_time = start_time if start_time is not None else ts_config.start_time
        actual_end_time = end_time if end_time is not None else ts_config.end_time
        actual_aggregation_type = aggregation_type if aggregation_type is not None else ts_config.aggregation_type
        actual_time_bucket = time_bucket if time_bucket is not None else ts_config.time_bucket
        actual_reverse = reverse_chronological if reverse_chronological is not None else ts_config.reverse_chronological

        if actual_start_time is None:
            actual_start_time = '-'  # Redis special value for earliest point

        if actual_end_time is None:
            actual_end_time = int(time.time() * 1000)  # Current time in milliseconds

        timestamps, values = self.redis_client.ts().revrange(
            key=key,
            from_time=actual_start_time,
            to_time=actual_end_time,
            count=actual_count,
            aggregation_type=actual_aggregation_type,
            bucket_size_msec=actual_time_bucket
        )

        data_points = [
            DataPoint(timestamp=ts, value=val)
            for ts, val in zip(timestamps, values)
        ]

        if not actual_reverse:
            data_points.reverse()

        return data_points

    def get_time_series(
            self,
            key: bytes | str | memoryview,
            config: Optional[TimeSeriesConfig] = None,
            **kwargs
    ) -> List[DataPoint]:
        """
        Fetch time series data from Redis with enhanced query options.

        Args:
            key: The Redis time series key
            config: TimeSeriesConfig object to use
            **kwargs: Optional arguments that override config values

        Returns:
            List of timestamps to values
        """
        # Determine which config to use, allowing override from kwargs
        ts_config = config or self.default_ts_config

        # Extract values from config
        aggregation_type = kwargs.get('aggregation_type', ts_config.aggregation_type)
        time_bucket = kwargs.get('time_bucket', ts_config.time_bucket)
        start_time = kwargs.get('start_time', ts_config.start_time)
        end_time = kwargs.get('end_time', ts_config.end_time)
        count = kwargs.get('count', ts_config.count)

        timestamps, values = self.redis_client.ts().range(
            key=key,
            from_time=start_time,
            to_time=end_time,
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
            config: Optional[AnomalyDetectionConfig] = None,
            **kwargs
    ) -> Tuple[List[AnomalyResult], Dict[str, Any]]:
        """
        Detect anomalies in a list of DataPoint objects using Luminol.

        Args:
            data_points: List of DataPoint objects containing timestamp and value
            config: AnomalyDetectionConfig object to use
            **kwargs: Optional arguments that override config values

        Returns:
            Tuple containing:
              - List of AnomalyResult objects with anomaly scores and classification
              - Dictionary with metadata about the detection process
        """
        # Determine which config to use
        anomaly_config = config or self.default_anomaly_config

        # Extract values from config, allowing override from kwargs
        threshold = kwargs.get('threshold', anomaly_config.threshold)
        algorithm = kwargs.get('algorithm', anomaly_config.algorithm)
        max_score = kwargs.get('max_score', anomaly_config.max_score)

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
            config: Optional[AnomalyDetectionConfig] = None,
            **kwargs
    ) -> Dict[str, Tuple[List[AnomalyResult], Dict]]:
        """
        Run anomaly detection with multiple algorithms and compare results.

        Args:
            data_points: List of DataPoint objects
            config: AnomalyDetectionConfig object to use
            **kwargs: Optional arguments that override config values

        Returns:
            Dictionary mapping algorithm names to their results and metadata
        """
        # Determine which config to use
        anomaly_config = config or self.default_anomaly_config

        # Extract values from config, allowing override from kwargs
        threshold = kwargs.get('threshold', anomaly_config.threshold)
        algorithms = kwargs.get('algorithms', anomaly_config.algorithms)

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
            config: Optional[AnomalyDetectionConfig] = None,
            **kwargs
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in a sliding window over a stream of data points.
        This approach is better for analyzing long time series by examining
        local patterns.

        Args:
            data_points: List of DataPoint objects
            config: AnomalyDetectionConfig object to use
            **kwargs: Optional arguments that override config values

        Returns:
            List of AnomalyResult objects for the entire dataset
        """
        # Determine which config to use
        anomaly_config = config or self.default_anomaly_config

        # Extract values from config, allowing override from kwargs
        window_size = kwargs.get('window_size', anomaly_config.window_size)
        threshold = kwargs.get('threshold', anomaly_config.threshold)
        algorithm = kwargs.get('algorithm', anomaly_config.algorithm)
        step_size = kwargs.get('step_size', anomaly_config.step_size)

        if len(data_points) < window_size:
            results, _ = self.detect_anomalies(data_points, threshold=threshold, algorithm=algorithm)
            return results

        all_results = []

        for i in range(0, len(data_points) - window_size + 1, step_size):
            window = data_points[i:i + window_size]
            window_results, _ = self.detect_anomalies(window, threshold=threshold, algorithm=algorithm)

            if i + step_size < len(data_points) - window_size + 1:
                window_results = window_results[:step_size]

            all_results.extend(window_results)

        return all_results
