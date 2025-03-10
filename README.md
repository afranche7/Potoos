# Potoos

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/00a1640c-5f44-4098-8aad-ff89a1b4455f" width="250" height="250">
</div>

Lightweight anomaly detection on redis time series data based on Luminol

## How to use
```python
from potoos.anomaly_detector import RedisTimeSeriesAnomalyDetector

detector = RedisTimeSeriesAnomalyDetector(redis_host='localhost', redis_port=6379)
time_series_data = detector.fetch_time_series('my_timeseries')
anomalies = detector.detect_anomalies(time_series_data)

for timestamp, score in anomalies.items():
    print(f"Anomaly detected at {timestamp}: Score = {score}")

```clau