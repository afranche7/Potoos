import redis
import luminol
import datetime


class RedisTimeSeriesAnomalyDetector:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

    def fetch_time_series(self, key, aggregation_type='avg', time_bucket=3600):
        """
        Fetch time series data from Redis.
        """
        timestamps, values = self.redis_client.ts().range(key, '-', '+', aggregation_type=aggregation_type,
                                                          time_bucket=time_bucket)

        time_series_data = {}
        for timestamp, value in zip(timestamps, values):
            time_series_data[datetime.datetime.fromtimestamp(timestamp / 1000)] = value

        return time_series_data

    def detect_anomalies(self, time_series_data):
        """
        Detect anomalies in the time series data using Luminol.
        """
        ts = luminol.time_series.TimeSeries(time_series_data)
        anomaly_detector = luminol.anomaly_detector.AnomalyDetector(ts)

        return anomaly_detector.get_anomalies()
