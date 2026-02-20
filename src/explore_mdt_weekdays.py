import pandas as pd
from pathlib import Path

# Reuse your loader pattern
def load_dates(path, date_cols):
    df = pd.read_csv(path)
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", format="%d/%m/%Y")
    return df

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def weekday_summary(df: pd.DataFrame, date_col: str, label: str):
    s = df[date_col].dropna()
    out = (
        s.dt.day_name()
        .value_counts()
        .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        .fillna(0)
        .astype(int)
    )
    props = (out / out.sum()).round(4)

    print(f"\n--- {label} ---")
    print("Counts by weekday:")
    print(out)
    print("\nProportions by weekday:")
    print(props)

    # Also return the numeric weekday set that actually occurs
    weekday_nums = sorted(s.dt.weekday.unique())  # 0=Mon..6=Sun
    print("\nWeekday numbers present (0=Mon..6=Sun):", weekday_nums)

   

    return out, props, weekday_nums

def main():
    # Biopsy / MRI MDT
    pre_mrirep_to_biopmdt = load_dates(
        DATA_DIR / "pre_mrirep_to_biopmdt.csv",
        ["Date MRI reported", "Date of Prostate MRI MDT"]
    )
    weekday_summary(
        pre_mrirep_to_biopmdt,
        "Date of Prostate MRI MDT",
        "Prostate MRI MDT (biopsy decision MDT)"
    )

    # Treatment options MDT
    pre_pathrep_to_treatmdt = load_dates(
        DATA_DIR / "pre_pathrep_to_treatmdt.csv",
        ["Date of pathology report", "Date of MDT (treatment options)"]
    )
    weekday_summary(
        pre_pathrep_to_treatmdt,
        "Date of MDT (treatment options)",
        "Treatment options MDT"
    )
     # explore why there are 2 days for MRI MDT

    df = pre_mrirep_to_biopmdt.copy()
    df = df.dropna(subset=["Date of Prostate MRI MDT"])

    df["year_month"] = df["Date of Prostate MRI MDT"].dt.to_period("M")
    df["weekday"] = df["Date of Prostate MRI MDT"].dt.day_name()

    pivot = (
        df.groupby(["year_month", "weekday"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nWeekday distribution by month:")
    print(pivot)

if __name__ == "__main__":
    main()