# Potoos

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/00a1640c-5f44-4098-8aad-ff89a1b4455f" width="250" height="250">
</div>

Lightweight anomaly detection on redis time series data based on Luminol

## How to use
```python
from redis import Redis
from potoos.client import PotoosClient, TimeSeriesConfig, AnomalyDetectionConfig


# 1. Initialize Redis client
redis_client = Redis(host='localhost', port=6379, db=0)

# 2. Create configuration objects
ts_config = TimeSeriesConfig(
    start_time='-1h',          # Get data from the last hour
    aggregation_type='avg',    # Average values
    time_bucket=60000          # Aggregate by minute (60,000 ms)
)

anomaly_config = AnomalyDetectionConfig(
    algorithm='derivative',    # Use derivative algorithm (good for trends)
    threshold=0.85,            # Higher threshold = fewer anomalies
    use_multiple_algorithms=False
)

# 3. Initialize PotoosClient
client = PotoosClient(
    redis_client=redis_client,
    default_ts_config=ts_config,
    default_anomaly_config=anomaly_config
)

# 4. Define a callback function to handle anomalies
def handle_anomalies(results, metadata):
    anomalies = [r for r in results if r.is_anomaly]
    if anomalies:
        print(f"Found {len(anomalies)} anomalies out of {len(results)} data points!")
        print(f"Anomaly timestamps: {[a.timestamp for a in anomalies]}")
        # Here you could send alerts, log to database, etc.
    else:
        print(f"No anomalies found among {len(results)} data points")
    
    print(f"Detection algorithm: {metadata['algorithm']}")
    print(f"Time range analyzed: {metadata['time_range']['duration_ms']/60000:.1f} minutes")

# 5. Monitor a Redis time series key
key = "system:cpu:usage"
results, metadata = client.monitor(
    key=key,
    callback=handle_anomalies
)
```