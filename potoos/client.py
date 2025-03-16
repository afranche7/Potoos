from redis.client import Redis
from luminol.anomaly_detector import AnomalyDetector
from luminol.modules.time_series import TimeSeries
from typing import List, Optional
from luminol.modules.anomaly import Anomaly
from .models.config import TimeSeriesConfig, AnomalyDetectionConfig
from .models.anomaly import DataPoint, AnomalyResult, MetaData, TimeRange


class PotoosClient:
    def __init__(
            self,
            redis_client: Redis,
            time_series_config: Optional[TimeSeriesConfig] = None,
            anomaly_config: Optional[AnomalyDetectionConfig] = None,
    ):
        self.redis_client: Redis = redis_client
        self.time_series_config: TimeSeriesConfig = time_series_config or TimeSeriesConfig()
        self.anomaly_config: AnomalyDetectionConfig = anomaly_config or AnomalyDetectionConfig()
        self._check_time_series_module()

    def _check_time_series_module(self):
        """
        Verifies that the RedisTimeSeries module is loaded on the Redis server.
        Raises an exception if the module is not available.
        """
        modules = self.redis_client.module_list()

        has_time_series = any(m.get('name') == 'timeseries' for m in modules)

        if not has_time_series:
            raise RuntimeError(
                "RedisTimeSeries module is not loaded on the Redis server. "
                "Please load the module before using time series functionality."
            )

    def _get_time_series(
            self,
            key: bytes | str | memoryview,
            config: Optional[TimeSeriesConfig] = None,
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
        ts_config: TimeSeriesConfig = config or self.time_series_config

        if ts_config.reversed:
            timestamps, values = self.redis_client.ts().revrange(key=key, **ts_config.__dict__)
        else:
            timestamps, values = self.redis_client.ts().range(key=key, **ts_config.__dict__)

        data_points: List[DataPoint] = []
        for timestamp, value in zip(timestamps, values):
            data_points.append(DataPoint(timestamp=timestamp, value=value))

        return data_points

    def _detect_anomalies(
            self,
            data_points: List[DataPoint],
    ) -> AnomalyResult:
        """
        Detect anomalies in a list of DataPoint objects using Luminol.

        Args:
            data_points: List of DataPoint objects containing timestamp and value
            anomaly_config: AnomalyDetectionConfig object to use
            **kwargs: Optional arguments that override config values

        Returns:
            Tuple containing:
              - List of AnomalyResult objects with anomaly scores and classification
              - Dictionary with metadata about the detection process
        """
        if len(data_points) < 4:
            raise ValueError("Not enough data points for anomaly detection (minimum 4 required)")

        data_dict = {point.timestamp / 1000: float(point.value) for point in data_points}

        time_series = TimeSeries(data_dict)

        detector = AnomalyDetector(time_series=time_series, **self.anomaly_config.__dict__)

        anomalies: List[Anomaly] = detector.get_anomalies()

        score_series = detector.get_all_scores()

        min_time: int = min(p.timestamp for p in data_points) if data_points else None
        max_time: int = max(p.timestamp for p in data_points) if data_points else None
        duration = max_time - min_time if max and min else None
        meta_data = MetaData(
            algorithm=self.anomaly_config.algorithm_name,
            data_points_analyzed=len(data_points),
            anomalies_found=len(anomalies),
            time_range_analyzed=TimeRange(
                start=min_time,
                end=max_time,
                duration=duration
            )
        )

        return AnomalyResult(anomalies=anomalies, scores=score_series, meta_data=meta_data)

    def monitor(
            self,
            key: bytes | str | memoryview,
            ts_config: Optional[TimeSeriesConfig] = None,
    ) -> Optional[AnomalyResult]:
        """
        Monitors a time series key by fetching data and performing anomaly detection.

        Args:
            key: The Redis time series key to monitor
            ts_config: Configuration for time series data retrieval

        Returns:
            Tuple containing:
                - List of AnomalyResult objects
                - Dictionary with metadata about the detection process
        """
        ts_config: TimeSeriesConfig = ts_config or self.time_series_config

        data_points: List[DataPoint] = self._get_time_series(key, config=ts_config)

        if not data_points:
            return None

        results = self._detect_anomalies(data_points=data_points)

        return results
