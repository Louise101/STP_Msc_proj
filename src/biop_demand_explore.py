import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "pre_biopmdt_to_biop.csv")

df["biopmdt_date"] = pd.to_datetime(df["Date of Prostate MRI MDT"], dayfirst=True)
df["biopsy_date"] = pd.to_datetime(df["Date of Biopsy"], dayfirst=True)

df["biopmdt_week"] = df["biopmdt_date"].dt.to_period("W").apply(lambda r: r.start_time)
df["biopsy_week"] = df["biopsy_date"].dt.to_period("W").apply(lambda r: r.start_time)

weekly_demand = df.groupby("biopmdt_week").size()
weekly_completed = df.groupby("biopsy_week").size()

weekly = pd.DataFrame({
    "demand": weekly_demand,
    "completed": weekly_completed
}).fillna(0)

print(weekly)