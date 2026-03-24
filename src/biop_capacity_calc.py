import pandas as pd
from pathlib import Path

def estimate_weekly_biopsy_capacity(df: pd.DataFrame, biopsy_date_col: str):
    df = df.copy()

    # ensure datetime
    df[biopsy_date_col] = pd.to_datetime(df[biopsy_date_col], dayfirst=True)

    # drop missing
    df = df.dropna(subset=[biopsy_date_col])

    # create week index
    df["week"] = df[biopsy_date_col].dt.to_period("W").apply(lambda r: r.start_time)

    weekly_counts = df.groupby("week").size()

    return weekly_counts

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "pre_biopmdt_to_biop.csv")

weekly_counts = estimate_weekly_biopsy_capacity(df, "Date of Biopsy")

mean_weekly_capacity = weekly_counts.mean()
median_weekly_capacity = weekly_counts.median()

print(weekly_counts)
print("Mean weekly capacity:", mean_weekly_capacity)
print("Median weekly capacity:", median_weekly_capacity)

weekly_capacity = int(round(median_weekly_capacity))

df["weekday"] = pd.to_datetime(df["Date of Biopsy"], dayfirst=True).dt.weekday

print(df["weekday"].value_counts().sort_index())

capacity_by_weekday = {
    3: 1,  # Thursday
    4: 1,  # Friday
}

import matplotlib.pyplot as plt

weekly_counts.plot(kind="hist", bins=10)
plt.title("Weekly biopsy counts")
plt.xlabel("Biopsies per week")
plt.show()