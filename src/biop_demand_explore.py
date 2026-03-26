import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "pre_biopmdt_to_biop.csv")

df["biopmdt_date"] = pd.to_datetime(df["Date of Prostate MRI MDT"], dayfirst=True)
df["biopsy_date"] = pd.to_datetime(df["Date of Biopsy"], dayfirst=True)

# Create weekly timeline
start = df["biopmdt_date"].min()
end = df["biopsy_date"].max()

weeks = pd.date_range(start=start, end=end, freq="W-MON")

results = []

for week in weeks:
    # New demand this week
    demand = df[
        (df["biopmdt_date"] >= week) &
        (df["biopmdt_date"] < week + pd.Timedelta(days=7))
    ].shape[0]

    # Completed this week
    completed = df[
        (df["biopsy_date"] >= week) &
        (df["biopsy_date"] < week + pd.Timedelta(days=7))
    ].shape[0]

    # Waiting (queue size)
    waiting = df[
        (df["biopmdt_date"] <= week) &
        (
            (df["biopsy_date"].isna()) |
            (df["biopsy_date"] > week)
        )
    ].shape[0]

    results.append({
        "week": week,
        "demand": demand,
        "completed": completed,
        "waiting": waiting,
    })

weekly = pd.DataFrame(results)

print(weekly)

weekly["prev_waiting"] = weekly["waiting"].shift(1)

print(weekly[["week", "prev_waiting", "completed"]].corr())

weekly["prev_waiting"] = weekly["waiting"].shift(1)

print("\n=== REAL DATA CORRELATIONS ===")
print("Completed vs demand:",
      weekly["completed"].corr(weekly["demand"]))

print("Completed vs previous waiting:",
      weekly["completed"].corr(weekly["prev_waiting"]))

print(np.median(weekly["completed"]))

