<p align="center">
  <img src="https://github.com/user-attachments/assets/5fdf1783-27dd-407e-bba6-bb1c809c689c" width="250" height="250">
</p>

# Potoos

Potoos is a lightweight Python library for time series anomaly detection using Redis Time Series and Luminol. Monitor your time series data for anomalies with minimal configuration.

## Overview

Potoos combines the power of:
- **[Redis Time Series](https://github.com/RedisTimeSeries/RedisTimeSeries)** for efficient time series data storage and retrieval
- **[Luminol](https://github.com/linkedin/luminol)** for robust anomaly detection algorithms

This makes it ideal for monitoring metrics, detecting unusual patterns, and identifying outliers in your time series data.

## Features

- 🔄 **Seamless Redis TimeSeries Integration**: Automatically verifies Redis module availability
- 🔍 **Configurable Time Series Retrieval**: Forward or reverse order, with flexible query options
- 🚨 **Anomaly Detection**: Uses Luminol's advanced algorithms to identify anomalies
- 🛠️ **Highly Configurable**: Customize both time series retrieval and anomaly detection parameters

## Requirements

- Python 3.10+
- Redis server with the RedisTimeSeries module installed
- Dependencies:
  - redis-py
  - luminol

## Installation

```bash
pip install potoos
```

Ensure your Redis server has the TimeSeries module installed:

```bash
# Check if module is installed
redis-cli MODULE LIST | grep timeseries

# If not found, install using Redis Stack or Redis modules
```

## Quick Start

Here's a simple example of how to use Potoos:

```python
from redis import Redis
from potoos.client import PotoosClient
from potoos.models.config import TimeSeriesConfig, AnomalyDetectionConfig

# Connect to Redis
redis_client = Redis(host='localhost', port=6379)

# Initialize Potoos client with default configurations
client = PotoosClient(redis_client)

# Or with custom configurations
client = PotoosClient(
    redis_client=redis_client,
    time_series_config=TimeSeriesConfig(reversed=False, count=1000),
    anomaly_config=AnomalyDetectionConfig(algorithm_name='bitmap_detector')
)

# Monitor a time series key for anomalies
results = client.monitor('metrics:cpu:usage')

# Process the results
if results:
    print(f"Analysis complete. Found {results.meta_data.anomalies_found} anomalies")
    print(f"Analyzed {results.meta_data.data_points_analyzed} data points")
    
    # Print information about each anomaly
    for anomaly in results.anomalies:
        print(f"Anomaly at {anomaly.exact_timestamp}")
        print(f"Anomaly score: {anomaly.anomaly_score}")
        
    # Or access anomaly score in TimeSeries object
    print(f"Anomaly scores: {results.scores}")

    # Access time range analyzed
    time_range = results.meta_data.time_range_analyzed
    print(f"Time range analyzed: {time_range.start} to {time_range.end}")
```

## Configuration

### Time Series Configuration

```python
from potoos.models.config import TimeSeriesConfig

# Default values shown
config = TimeSeriesConfig(
    count=None,         # Maximum number of samples to return
    aggregation=None,   # Aggregation type (e.g., 'avg', 'sum', 'min', 'max')
    bucket_size=None,   # Time bucket for aggregation in milliseconds
    filter_by=None,     # Filtering options for labels
    align=None,         # Timestamp alignment control
    start=None,         # Start timestamp
    end=None,           # End timestamp
    reversed=False      # Return results in reverse order when True
)
```

### Anomaly Detection Configuration

```python
from potoos.models.config import AnomalyDetectionConfig

# Default values shown
config = AnomalyDetectionConfig(
    algorithm_name='bitmap_detector',  # Algorithm to use
    score_threshold=None,              # Threshold for anomaly detection
    score_percentile_threshold=None,   # Percentile threshold
    algorithm_params={}                # Additional algorithm parameters
)
```

## How It Works

1. **Initialization**: PotoosClient connects to your Redis instance and verifies the TimeSeries module is available
2. **Data Retrieval**: When monitoring, it fetches time series data according to your configuration
3. **Anomaly Detection**: The retrieved data is analyzed using Luminol's algorithms
4. **Results**: You receive detailed information about detected anomalies and analysis metadata

## Note on Dependencies

Potoos requires NumPy 1.22.4 or earlier due to Luminol's dependency on the `numpy.asscalar()` function, which was removed in later versions of NumPy.
