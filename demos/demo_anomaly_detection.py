"""WavqWise Anomaly Detection Demo"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wavqwise import AnomalyPipeline

# Load sensor data with anomalies
pipeline = AnomalyPipeline()
pipeline.load("demos/sample_data/sensor_readings.csv", target="temperature", time="timestamp")

# Detect with Z-score
print("=== Z-Score Detection ===")
result = pipeline.detect(method="zscore")
print(result.summary())

# Detect with IQR
print("\n=== IQR Detection ===")
result_iqr = pipeline.detect(method="iqr")
print(result_iqr.summary())

# Show anomaly details
print(f"\nTop 5 anomalies:")
top = result.anomalies.nlargest(5, "anomaly_score")
print(top[["timestamp", "temperature", "anomaly_score", "severity"]].to_string(index=False))
