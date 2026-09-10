import matplotlib.pyplot as plt
from metrics_source import SyntheticMetricSource, SyntheticConfig

# Generate 3 hours of data
cfg = SyntheticConfig(duration_minutes=180, burst_probability=0.02)
src = SyntheticMetricSource(cfg)

df = src.generate_dataset()

# Convert timestamp to minutes from start
time_minutes = (
    df["timestamp"] - df["timestamp"].iloc[0]
).dt.total_seconds() / 60

# Plot request rate
plt.figure(figsize=(12, 5))
plt.plot(time_minutes, df["request_rate"])
plt.xlabel("Time (minutes)")
plt.ylabel("Request rate")
plt.title("Synthetic Request Rate")
plt.grid(True)
plt.show()

# Plot CPU and memory
plt.figure(figsize=(12, 5))
plt.plot(time_minutes, df["cpu_pct"], label="CPU %")
plt.plot(time_minutes, df["mem_pct"], label="Memory %")
plt.xlabel("Time (minutes)")
plt.ylabel("Utilization (%)")
plt.title("CPU and Memory vs Time")
plt.legend()
plt.grid(True)
plt.show()