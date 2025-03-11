import unittest
from unittest.mock import Mock, patch, MagicMock
from potoos.client import PotoosClient, TimeSeriesConfig, AnomalyDetectionConfig, AnomalyResult


class TestPotoosClient(unittest.TestCase):
    """Test suite for PotoosClient's monitor function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock Redis client
        self.redis_client = Mock()

        # Mock the module_list method to indicate RedisTimeSeries is available
        self.redis_client.module_list.return_value = [
            {"name": "timeseries", "ver": 10205}
        ]

        # Create a mock for the Redis time series functionality
        self.ts_mock = Mock()
        self.redis_client.ts.return_value = self.ts_mock

        # Sample time series data (timestamps and values)
        self.sample_ts_data = [
            (1614556800000, 10.5),  # Normal data
            (1614557700000, 11.2),  # Normal data
            (1614558600000, 10.8),  # Normal data
            (1614559500000, 12.1),  # Normal data
            (1614560400000, 11.5),  # Normal data
            (1614561300000, 32.7),  # Anomaly
            (1614562200000, 11.9),  # Normal data
            (1614563100000, 12.3),  # Normal data
            (1614564000000, 11.7),  # Normal data
            (1614564900000, 12.0),  # Normal data
        ]

        # Set up the mock to return our sample data
        self.ts_mock.range.return_value = list(zip(*self.sample_ts_data))
        self.ts_mock.revrange.return_value = list(zip(*reversed(self.sample_ts_data)))

        # Initialize the client with default configs
        self.client = PotoosClient(self.redis_client)

        # Create a spy for detect_anomalies method
        self.original_detect_anomalies = self.client.detect_anomalies
        self.client.detect_anomalies = MagicMock(wraps=self.original_detect_anomalies)

        # Sample anomaly results for mocking
        self.sample_anomaly_results = [
            AnomalyResult(timestamp=ts, value=val,
                          anomaly_score=0.9 if val > 30 else 0.1,
                          is_anomaly=val > 30)
            for ts, val in self.sample_ts_data
        ]

        # Sample metadata for mocking
        self.sample_metadata = {
            'algorithm': 'derivative',
            'threshold': 0.8,
            'data_points_analyzed': len(self.sample_ts_data),
            'anomalies_found': 1,
            'max_score': 0.9,
        }

    def test_monitor_basic_functionality(self):
        """Test that monitor correctly fetches data and detects anomalies."""
        # Set up the detect_anomalies mock to return our sample results
        self.client.detect_anomalies.return_value = (self.sample_anomaly_results, self.sample_metadata)

        # Call the monitor function
        results, metadata = self.client.monitor(key='test:key')

        # Assert that the ts.range method was called correctly
        self.ts_mock.range.assert_called_once()

        # Assert that detect_anomalies was called with the correct data
        self.client.detect_anomalies.assert_called_once()

        # Verify the results
        self.assertEqual(len(results), len(self.sample_anomaly_results))
        self.assertEqual(metadata['anomalies_found'], 1)

        # Check that the anomaly was detected at the correct timestamp
        anomalies = [r for r in results if r.is_anomaly]
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].timestamp, 1614561300000)
        self.assertEqual(anomalies[0].value, 32.7)

    def test_monitor_with_custom_configs(self):
        """Test monitor with custom time series and anomaly configurations."""
        # Create custom configs
        ts_config = TimeSeriesConfig(
            start_time='-1h',
            end_time='+',
            count=50,
            aggregation_type='avg',
            time_bucket=60000
        )

        anomaly_config = AnomalyDetectionConfig(
            algorithm='exp_avg_detector',
            threshold=0.75
        )

        # Set up the detect_anomalies mock
        self.client.detect_anomalies.return_value = (self.sample_anomaly_results, self.sample_metadata)

        # Call monitor with custom configs
        self.client.monitor(
            key='test:key',
            ts_config=ts_config,
            anomaly_config=anomaly_config
        )

        # Assert that the ts.range method was called with the correct parameters
        self.ts_mock.range.assert_called_once_with(
            key='test:key',
            from_time='-1h',
            to_time='+',
            aggregation_type='avg',
            time_bucket=60000,
            count=50
        )

        # Get the arguments passed to detect_anomalies
        _, kwargs = self.client.detect_anomalies.call_args

        # Verify the anomaly config was used
        self.assertEqual(kwargs.get('algorithm'), 'exp_avg_detector')
        self.assertEqual(kwargs.get('threshold'), 0.75)

    def test_monitor_with_callback(self):
        """Test that the callback function is called with the correct results."""
        # Create a mock callback function
        callback_mock = Mock()

        # Set up the detect_anomalies mock
        self.client.detect_anomalies.return_value = (self.sample_anomaly_results, self.sample_metadata)

        # Call monitor with the callback
        self.client.monitor(key='test:key', callback=callback_mock)

        # Assert that the callback was called with the correct arguments
        callback_mock.assert_called_once_with(self.sample_anomaly_results, self.sample_metadata)

    def test_monitor_with_multiple_algorithms(self):
        """Test monitor when using multiple anomaly detection algorithms."""
        # Create an anomaly config with multiple algorithms
        anomaly_config = AnomalyDetectionConfig(
            use_multiple_algorithms=True,
            algorithms=['derivative', 'exp_avg_detector'],
            algorithm='derivative'  # Primary algorithm
        )

        # Create a mock for detect_anomalies_multiple_algorithms
        self.client.detect_anomalies_multiple_algorithms = Mock()

        # Set up the mock to return a dictionary of results
        multi_algo_results = {
            'derivative': (self.sample_anomaly_results, self.sample_metadata),
            'exp_avg_detector': (
                [AnomalyResult(timestamp=ts, value=val, anomaly_score=0.8 if val > 30 else 0.2, is_anomaly=val > 30)
                 for ts, val in self.sample_ts_data],
                {'algorithm': 'exp_avg_detector', 'anomalies_found': 1}
            )
        }
        self.client.detect_anomalies_multiple_algorithms.return_value = multi_algo_results

        # Call monitor with the config using multiple algorithms
        results, metadata = self.client.monitor(
            key='test:key',
            anomaly_config=anomaly_config
        )

        # Assert that detect_anomalies_multiple_algorithms was called
        self.client.detect_anomalies_multiple_algorithms.assert_called_once()

        # Verify that we got the results from the primary algorithm
        self.assertEqual(results, self.sample_anomaly_results)
        self.assertEqual(metadata['algorithm'], 'derivative')

        # Verify that the multi-algorithm results are in the metadata
        self.assertIn('multi_algorithm_results', metadata)
        self.assertEqual(metadata['multi_algorithm_results'], multi_algo_results)

    def test_monitor_with_stream_detection(self):
        """Test monitor when using stream-based anomaly detection."""
        # Create an anomaly config with stream detection
        anomaly_config = AnomalyDetectionConfig(
            use_stream_detection=True,
            window_size=5,
            step_size=1,
            algorithm='derivative'
        )

        # Create a mock for detect_and_analyze_stream
        self.client.detect_and_analyze_stream = Mock()
        self.client.detect_and_analyze_stream.return_value = self.sample_anomaly_results

        # Call monitor with the stream detection config
        results, metadata = self.client.monitor(
            key='test:key',
            anomaly_config=anomaly_config
        )

        # Assert that detect_and_analyze_stream was called
        self.client.detect_and_analyze_stream.assert_called_once()

        # Verify the results
        self.assertEqual(results, self.sample_anomaly_results)
        self.assertEqual(metadata['detection_method'], 'stream')
        self.assertEqual(metadata['window_size'], 5)
        self.assertEqual(metadata['step_size'], 1)

    def test_monitor_with_no_data(self):
        """Test monitor behavior when no data is returned from Redis."""
        # Set up the ts.range mock to return empty lists
        self.ts_mock.range.return_value = ([], [])

        # Call monitor
        results, metadata = self.client.monitor(key='test:key')

        # Verify that we got empty results and an error in metadata
        self.assertEqual(results, [])
        self.assertIn('error', metadata)
        self.assertEqual(metadata['error'], 'No data points retrieved')

        # Assert that detect_anomalies was not called
        self.client.detect_anomalies.assert_not_called()

    def test_monitor_with_real_data_conversion(self):
        """Test that monitor correctly converts Redis data to DataPoint objects."""
        # Reset the detect_anomalies mock to use the actual implementation
        self.client.detect_anomalies = self.original_detect_anomalies

        # Create a patch for AnomalyDetector to control its behavior
        with patch('luminol.anomaly_detector.AnomalyDetector') as mock_detector_class:
            # Create a mock detector instance
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector

            # Set up the mock detector to return sample anomalies and scores
            mock_detector.get_anomalies.return_value = []
            mock_detector.get_all_scores.return_value = {
                ts / 1000: 0.9 if val > 30 else 0.1
                for ts, val in self.sample_ts_data
            }

            # Call monitor
            results, metadata = self.client.monitor(key='test:key')

            # Verify that TimeSeries was created with the correct data
            # This is a bit tricky to test directly, so we'll check that AnomalyDetector was initialized
            mock_detector_class.assert_called_once()

            # Verify that the results contain the expected number of points
            self.assertEqual(len(results), len(self.sample_ts_data))

            # Check that the anomaly was detected at the correct timestamp
            anomalies = [r for r in results if r.is_anomaly]
            self.assertEqual(len(anomalies), 1)
            self.assertEqual(anomalies[0].value, 32.7)

    def test_monitor_handles_exceptions(self):
        """Test that monitor properly handles exceptions during data processing."""
        # Make ts.range raise an exception
        self.ts_mock.range.side_effect = Exception("Redis connection error")

        # Ensure monitor handles this gracefully
        with self.assertRaises(Exception) as context:
            self.client.monitor(key='test:key')

        self.assertIn("Redis connection error", str(context.exception))


if __name__ == '__main__':
    unittest.main()