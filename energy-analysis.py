import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("household-power-consumption.csv")

# Normalize column names
df.columns = df.columns.str.lower().str.strip()

print("Columns found in dataset:")
print(df.columns)

# -----------------------------
# 2. Detect Datetime Column
# -----------------------------
datetime_candidates = [c for c in df.columns if "date" in c or "time" in c]
if not datetime_candidates:
    raise Exception("No datetime column found in dataset")

datetime_col = datetime_candidates[0]
df[datetime_col] = pd.to_datetime(df[datetime_col])
df.set_index(datetime_col, inplace=True)

# -----------------------------
# 3. Detect Power / Consumption Column
# -----------------------------
power_candidates = [
    c for c in df.columns
    if "power" in c or "consumption" in c or "energy" in c
]

if not power_candidates:
    raise Exception("No power/consumption column found")

power_col = power_candidates[0]
df[power_col] = pd.to_numeric(df[power_col], errors="coerce")
df.dropna(inplace=True)

# -----------------------------
# 4. Resample to Hourly Data
# -----------------------------
hourly_data = df[power_col].resample("H").mean()

hourly_df = hourly_data.to_frame(name="Load")

# -----------------------------
# 5. Peak vs Off-Peak Analysis
# -----------------------------
peak_threshold = hourly_df["Load"].quantile(0.75)
hourly_df["Peak_Type"] = np.where(
    hourly_df["Load"] >= peak_threshold,
    "Peak",
    "Off-Peak"
)

# -----------------------------
# 6. Rolling Average Trend
# -----------------------------
hourly_df["Rolling_24H"] = hourly_df["Load"].rolling(24).mean()

# -----------------------------
# 7. Weekday vs Weekend
# -----------------------------
hourly_df["Day_Type"] = np.where(
    hourly_df.index.dayofweek < 5,
    "Weekday",
    "Weekend"
)

weekday_weekend_avg = hourly_df.groupby("Day_Type")["Load"].mean()

# -----------------------------
# 8. Anomaly Detection
# -----------------------------
mean_load = hourly_df["Load"].mean()
std_load = hourly_df["Load"].std()

hourly_df["Anomaly"] = np.where(
    (hourly_df["Load"] > mean_load + 2 * std_load) |
    (hourly_df["Load"] < mean_load - 2 * std_load),
    1, 0
)

# -----------------------------
# 9. Energy Efficiency Score
# -----------------------------
max_load = hourly_df["Load"].max()
hourly_df["Efficiency_Score"] = 100 - (hourly_df["Load"] / max_load * 100)
hourly_df["Efficiency_Score"] = hourly_df["Efficiency_Score"].clip(0, 100)

# -----------------------------
# 10. Visualizations
# -----------------------------
plt.figure()
plt.plot(hourly_df.index, hourly_df["Load"], label="Load")
plt.plot(hourly_df.index, hourly_df["Rolling_24H"], label="Rolling Avg")
plt.legend()
plt.title("Hourly Load Trend")
plt.show()

plt.figure()
weekday_weekend_avg.plot(kind="bar")
plt.title("Weekday vs Weekend Consumption")
plt.show()

pivot = hourly_df.copy()
pivot["Hour"] = pivot.index.hour
pivot["Day"] = pivot.index.date

heatmap = pivot.pivot_table(
    values="Load", index="Hour", columns="Day", aggfunc="mean"
)

plt.figure()
plt.imshow(heatmap, aspect="auto")
plt.colorbar(label="Load")
plt.title("Hourly Load Heatmap")
plt.xlabel("Day")
plt.ylabel("Hour")
plt.show()

# -----------------------------
# 11. Results Summary
# -----------------------------
print("Peak Load Threshold:", peak_threshold)
print("Total Anomalies Detected:", hourly_df["Anomaly"].sum())
print("Average Energy Efficiency Score:",
      hourly_df["Efficiency_Score"].mean())