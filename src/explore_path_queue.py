from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(DATA_DIR / "pre_biop_to_pathrep.csv")

# Adjust column names if needed
df["biopsy_date"] = pd.to_datetime(df["Date of Biopsy"], dayfirst=True)
df["pathrep_date"] = pd.to_datetime(df["Date of pathology report"], dayfirst=True)

# Drop missing
df = df.dropna(subset=["biopsy_date", "pathrep_date"])

# ---------------------------
# Build weekly demand (biopsy)
# ---------------------------
df["biopsy_week"] = df["biopsy_date"].dt.to_period("W").apply(lambda r: r.start_time)
df["pathrep_week"] = df["pathrep_date"].dt.to_period("W").apply(lambda r: r.start_time)

weekly_demand = df.groupby("biopsy_week").size().rename("demand")
weekly_completed = df.groupby("pathrep_week").size().rename("completed")

# Combine
weekly = pd.concat([weekly_demand, weekly_completed], axis=1).fillna(0).sort_index()

# Ensure ints
weekly["demand"] = weekly["demand"].astype(int)
weekly["completed"] = weekly["completed"].astype(int)

# ---------------------------
# Build backlog (queue proxy)
# ---------------------------
weekly["waiting"] = weekly["demand"].cumsum() - weekly["completed"].cumsum()
weekly["prev_waiting"] = weekly["waiting"].shift(1)

# ---------------------------
# Print basic summary
# ---------------------------
print("\n=== Weekly pathology summary ===")
print(weekly.head(20))

print("\n=== Demand summary ===")
print(weekly["demand"].describe())

print("\n=== Completed summary ===")
print(weekly["completed"].describe())

# ---------------------------
# Correlations
# ---------------------------
corr = weekly[["demand", "completed", "prev_waiting"]].corr()

print("\n=== Correlations ===")
print(corr)

print("Completed vs demand:", corr.loc["completed", "demand"])
print("Completed vs previous waiting:", corr.loc["completed", "prev_waiting"])

# ---------------------------
# Plot 1: backlog over time
# ---------------------------
plt.figure()
plt.plot(weekly.index, weekly["waiting"])
plt.xticks(rotation=45)
plt.title("Pathology backlog (proxy)")
plt.xlabel("Week")
plt.ylabel("Waiting")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pathology_backlog.png", dpi=300)
plt.close()

# ---------------------------
# Plot 2: demand vs completed
# ---------------------------
plt.figure()
plt.plot(weekly.index, weekly["demand"], label="Demand (biopsy)")
plt.plot(weekly.index, weekly["completed"], label="Completed (path reports)")
plt.xticks(rotation=45)
plt.legend()
plt.title("Pathology demand vs completed")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pathology_demand_vs_completed.png", dpi=300)
plt.close()

# ---------------------------
# Plot 3: backlog vs completed
# ---------------------------
plt.figure()
plt.scatter(weekly["prev_waiting"], weekly["completed"])
plt.xlabel("Previous backlog")
plt.ylabel("Completed")
plt.title("Pathology capacity response to backlog")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pathology_backlog_vs_completed.png", dpi=300)
plt.close()

# ---------------------------
# Wait time distribution
# ---------------------------
df["wait_days"] = (df["pathrep_date"] - df["biopsy_date"]).dt.days

print("\n=== Wait time summary ===")
print(df["wait_days"].describe())

plt.figure()
plt.hist(df["wait_days"], bins=20)
plt.title("Biopsy to pathology report wait distribution")
plt.xlabel("Days")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pathology_wait_hist.png", dpi=300)
plt.close()


# ---------------------------
# Weekday exploration
# ---------------------------
weekday_map = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}

df["pathrep_weekday_num"] = df["pathrep_date"].dt.weekday
df["pathrep_weekday"] = df["pathrep_weekday_num"].map(weekday_map)

weekday_counts = (
    df["pathrep_weekday"]
    .value_counts()
    .reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fill_value=0)
)

weekday_props = (weekday_counts / weekday_counts.sum()).round(3)

weekday_summary = pd.DataFrame({
    "count": weekday_counts,
    "proportion": weekday_props,
})

print("\n=== Pathology report weekday summary ===")
print(weekday_summary)

# Optional: weekday counts by week, to see if certain weekdays are repeatedly used
weekly_weekday_counts = (
    df.groupby([df["pathrep_date"].dt.to_period("W").apply(lambda r: r.start_time), "pathrep_weekday"])
      .size()
      .unstack(fill_value=0)
)

print("\n=== Weekly pathology report counts by weekday ===")
print(weekly_weekday_counts.head(15))

# Plot weekday counts
plt.figure()
weekday_counts.plot(kind="bar")
plt.title("Pathology reports by weekday")
plt.xlabel("Weekday")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pathology_report_weekday_counts.png", dpi=300)
plt.close()


df["biopsy_weekday"] = df["biopsy_date"].dt.weekday.map(weekday_map)

biopsy_weekday_counts = (
    df["biopsy_weekday"]
    .value_counts()
    .reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fill_value=0)
)

biopsy_vs_report_weekday = pd.DataFrame({
    "biopsy_count": biopsy_weekday_counts,
    "pathrep_count": weekday_counts,
})

print("\n=== Biopsy vs pathology weekday counts ===")
print(biopsy_vs_report_weekday)
# ---------------------------
# Save data
# ---------------------------
weekly.to_csv(OUTPUT_DIR / "pathology_weekly.csv")
df[["biopsy_date", "pathrep_date", "wait_days"]].to_csv(
    OUTPUT_DIR / "pathology_waits.csv", index=False
)

print("\nSaved outputs to:", OUTPUT_DIR)
print("\n=== Pathology weekday numeric summary ===")
print(df["pathrep_date"].dt.weekday.value_counts().sort_index())