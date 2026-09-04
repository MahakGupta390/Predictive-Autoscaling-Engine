import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/fabricated_workload.csv", parse_dates=["timestamp"])
df.plot(x="timestamp", y="cpu_pct")
plt.savefig("data/cpu_preview.png")
print("Saved preview to data/cpu_preview.png")