import unittest
from potoos.anomaly_detector import RedisTimeSeriesAnomalyDetector


class TestAnomalyDetection(unittest.TestCase):
    def test_anomalies(self):
        detector = RedisTimeSeriesAnomalyDetector(redis_host='localhost', redis_port=6379)

        # Use a mock or real Redis instance, e.g., "my_timeseries"
        time_series_data = detector.fetch_time_series('my_timeseries')

        anomalies = detector.detect_anomalies(time_series_data)

        # Test that anomalies were detected
        self.assertGreater(len(anomalies), 0)


if __name__ == '__main__':
    unittest.main()
