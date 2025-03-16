import unittest
from unittest.mock import Mock, patch
from redis.client import Redis
from luminol.anomaly_detector import AnomalyDetector
from luminol.modules.time_series import TimeSeries
from luminol.modules.anomaly import Anomaly
from potoos.client import PotoosClient
from potoos.models.config import TimeSeriesConfig, AnomalyDetectionConfig
from potoos.models.anomaly import DataPoint, AnomalyResult, MetaData, TimeRange


class TestPotoosClient(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock Redis client
        self.mock_redis = Mock(spec=Redis)

        # Mock the module_list method to return a list with the timeseries module
        self.mock_redis.module_list.return_value = [{'name': 'timeseries'}]

        # Mock the ts() method
        self.mock_ts = Mock()
        self.mock_redis.ts.return_value = self.mock_ts

        # Default configurations
        self.time_series_config = TimeSeriesConfig()
        self.anomaly_config = AnomalyDetectionConfig()

        # Create client with mocked dependencies
        self.client = PotoosClient(
            redis_client=self.mock_redis,
            time_series_config=self.time_series_config,
            anomaly_config=self.anomaly_config
        )

    # 1. Basic initialization tests
    def test_init_with_defaults(self):
        """Test client initialization with default configurations."""
        # Reset the mock to clear previous calls
        self.mock_redis.reset_mock()

        client = PotoosClient(redis_client=self.mock_redis)

        self.assertIsInstance(client.time_series_config, TimeSeriesConfig)
        self.assertIsInstance(client.anomaly_config, AnomalyDetectionConfig)
        self.mock_redis.module_list.assert_called_once()

    def test_init_with_custom_configs(self):
        """Test client initialization with custom configurations."""
        custom_ts_config = TimeSeriesConfig(reversed=True)
        custom_anomaly_config = AnomalyDetectionConfig(algorithm_name="derivative_algorithm")

        client = PotoosClient(
            redis_client=self.mock_redis,
            time_series_config=custom_ts_config,
            anomaly_config=custom_anomaly_config
        )

        self.assertEqual(client.time_series_config, custom_ts_config)
        self.assertEqual(client.anomaly_config, custom_anomaly_config)

    # 2. Redis module checking tests
    def test_module_check_success(self):
        """Test successful module check when timeseries module is present."""
        # Already set up in setUp method, just verify it works
        client = PotoosClient(redis_client=self.mock_redis)
        # No exception should be raised

    def test_module_check_failure(self):
        """Test module check failure when timeseries module is missing."""
        self.mock_redis.module_list.return_value = [{'name': 'other_module'}]

        with self.assertRaises(RuntimeError):
            PotoosClient(redis_client=self.mock_redis)

    # 3. Time series data retrieval tests
    def test_get_time_series_forward(self):
        """Test retrieving time series data in forward order."""
        key = "test_key"
        self.mock_ts.range.return_value = ([1000, 2000, 3000], [10.0, 20.0, 30.0])

        result = self.client._get_time_series(key)

        self.mock_ts.range.assert_called_once_with(key=key, **self.time_series_config.__dict__)
        expected = [
            DataPoint(timestamp=1000, value=10.0),
            DataPoint(timestamp=2000, value=20.0),
            DataPoint(timestamp=3000, value=30.0)
        ]
        self.assertEqual(result, expected)

    def test_get_time_series_reversed(self):
        """Test retrieving time series data in reverse order."""
        key = "test_key"
        reversed_config = TimeSeriesConfig(reversed=True)
        self.mock_ts.revrange.return_value = ([3000, 2000, 1000], [30.0, 20.0, 10.0])

        result = self.client._get_time_series(key, config=reversed_config)

        self.mock_ts.revrange.assert_called_once_with(key=key, **reversed_config.__dict__)
        expected = [
            DataPoint(timestamp=3000, value=30.0),
            DataPoint(timestamp=2000, value=20.0),
            DataPoint(timestamp=1000, value=10.0)
        ]
        self.assertEqual(result, expected)

    def test_get_time_series_empty(self):
        """Test retrieving time series data when no data is available."""
        key = "empty_key"
        self.mock_ts.range.return_value = ([], [])

        result = self.client._get_time_series(key)

        self.assertEqual(result, [])

    def test_get_time_series_with_different_key_types(self):
        """Test retrieving time series with different key types (str, bytes, memoryview)."""
        # String key
        str_key = "string_key"
        self.mock_ts.range.return_value = ([1000], [10.0])
        self.client._get_time_series(str_key)
        self.mock_ts.range.assert_called_with(key=str_key, **self.time_series_config.__dict__)

        # Bytes key
        bytes_key = b"bytes_key"
        self.mock_ts.range.return_value = ([1000], [10.0])
        self.client._get_time_series(bytes_key)
        self.mock_ts.range.assert_called_with(key=bytes_key, **self.time_series_config.__dict__)

    # 4. Anomaly detection tests
    def test_detect_anomalies_insufficient_data(self):
        """Test anomaly detection with insufficient data points."""
        data_points = [
            DataPoint(timestamp=1000, value=10.0),
            DataPoint(timestamp=2000, value=20.0),
            DataPoint(timestamp=3000, value=30.0)
        ]  # Only 3 data points, need at least 4

        with self.assertRaises(ValueError):
            self.client._detect_anomalies(data_points)

    def test_detect_anomalies_basic(self):
        """Test basic anomaly detection functionality."""
        # Sample data points
        data_points = [
            DataPoint(timestamp=1000, value=10.0),
            DataPoint(timestamp=2000, value=20.0),
            DataPoint(timestamp=3000, value=30.0),
            DataPoint(timestamp=4000, value=100.0)  # Potential anomaly
        ]

        # Create a proper mock of TimeSeries for the scores
        mock_scores = Mock(spec=TimeSeries)
        # Configure the TimeSeries mock to have the necessary properties
        mock_scores.timestamps = [1000, 2000, 3000, 4000]
        mock_scores.values = [0.1, 0.2, 0.3, 0.9]

        # Create a mock anomaly
        mock_anomaly = Mock(spec=Anomaly)

        # Use context manager to patch only the specific methods
        with patch.object(AnomalyDetector, 'get_anomalies', return_value=[mock_anomaly]), \
                patch.object(AnomalyDetector, 'get_all_scores', return_value=mock_scores):
            # Call the method
            result = self.client._detect_anomalies(data_points)

            # Check result structure
            self.assertEqual(len(result.anomalies), 1)
            self.assertEqual(result.anomalies[0], mock_anomaly)
            self.assertEqual(result.scores, mock_scores)
            self.assertEqual(result.meta_data.algorithm, self.anomaly_config.algorithm_name)
            self.assertEqual(result.meta_data.data_points_analyzed, 4)
            self.assertEqual(result.meta_data.anomalies_found, 1)
            self.assertEqual(result.meta_data.time_range_analyzed.start, 1000)
            self.assertEqual(result.meta_data.time_range_analyzed.end, 4000)
            self.assertEqual(result.meta_data.time_range_analyzed.duration, 3000)

    def test_detect_anomalies_no_anomalies(self):
        """Test anomaly detection when no anomalies are found."""
        data_points = [
            DataPoint(timestamp=1000, value=10.0),
            DataPoint(timestamp=2000, value=20.0),
            DataPoint(timestamp=3000, value=30.0),
            DataPoint(timestamp=4000, value=40.0)
        ]

        # Create a proper mock of TimeSeries for the scores
        mock_scores = Mock(spec=TimeSeries)
        # Configure the TimeSeries mock to have the necessary properties
        mock_scores.timestamps = [1000, 2000, 3000, 4000]
        mock_scores.values = [0.1, 0.1, 0.1, 0.1]

        # Use a context manager to patch the AnomalyDetector methods
        with patch.object(AnomalyDetector, 'get_anomalies', return_value=[]), \
                patch.object(AnomalyDetector, 'get_all_scores', return_value=mock_scores):
            # Call the method
            result = self.client._detect_anomalies(data_points)

            # Check expected results
            self.assertEqual(len(result.anomalies), 0)
            self.assertEqual(result.scores, mock_scores)
            self.assertEqual(result.meta_data.anomalies_found, 0)

    # 5. End-to-end monitoring tests
    def test_monitor_with_no_data(self):
        """Test monitor method when no data is available."""
        key = "no_data_key"
        self.mock_ts.range.return_value = ([], [])

        result = self.client.monitor(key)

        self.assertIsNone(result)
        self.mock_ts.range.assert_called_once_with(key=key, **self.time_series_config.__dict__)

    def test_monitor_basic(self):
        """Test basic monitoring functionality."""
        key = "test_key"
        self.mock_ts.range.return_value = ([1000, 2000, 3000, 4000], [10.0, 20.0, 30.0, 40.0])

        # Mock the anomaly detection
        with patch.object(self.client, '_detect_anomalies') as mock_detect:
            mock_result = Mock(spec=AnomalyResult)
            mock_detect.return_value = mock_result

            result = self.client.monitor(key)

            # Verify the calls
            self.mock_ts.range.assert_called_once_with(key=key, **self.time_series_config.__dict__)
            mock_detect.assert_called_once()

            # Check data points passed to detect_anomalies
            data_points_arg = mock_detect.call_args[1]['data_points']
            self.assertEqual(len(data_points_arg), 4)

            # Verify result
            self.assertEqual(result, mock_result)

    def test_monitor_with_custom_ts_config(self):
        """Test monitoring with a custom time series configuration."""
        key = "test_key"
        custom_config = TimeSeriesConfig(reversed=True)

        self.mock_ts.revrange.return_value = ([4000, 3000, 2000, 1000], [40.0, 30.0, 20.0, 10.0])

        # Mock the anomaly detection
        with patch.object(self.client, '_detect_anomalies') as mock_detect:
            mock_result = Mock(spec=AnomalyResult)
            mock_detect.return_value = mock_result

            result = self.client.monitor(key, ts_config=custom_config)

            # Verify the calls
            self.mock_ts.revrange.assert_called_once_with(key=key, **custom_config.__dict__)
            self.assertEqual(result, mock_result)

    def test_monitor_error_propagation(self):
        """Test that monitor properly propagates errors from underlying methods."""
        key = "test_key"
        self.mock_ts.range.return_value = ([1000, 2000, 3000, 4000], [10.0, 20.0, 30.0, 40.0])

        # Mock the anomaly detection to raise an exception
        with patch.object(self.client, '_detect_anomalies') as mock_detect:
            mock_detect.side_effect = Exception("Test error")

            with self.assertRaises(Exception):
                self.client.monitor(key)


if __name__ == '__main__':
    unittest.main()