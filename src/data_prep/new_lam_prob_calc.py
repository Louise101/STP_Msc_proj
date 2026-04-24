import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

pre_df = pd.read_csv(DATA_DIR / "pre_refferal.csv").copy()
pros_df = pd.read_csv(DATA_DIR / "pros_ref.csv").copy()

# Parse dates
pre_df["referral_date"] = pd.to_datetime(
    pre_df["Date of referral to pathway"],
    dayfirst=True,
    errors="coerce",
)

pros_df["referral_date"] = pd.to_datetime(
    pros_df["Date of referral to pathway"],
    format="%m/%d/%y",
    errors="coerce",
)

# Keep only valid dates
pre_df = pre_df[pre_df["referral_date"].notna()].copy()
pros_df = pros_df[pros_df["referral_date"].notna()].copy()

# Add labels
pre_df["pathway"] = "BASELINE"
pros_df["pathway"] = "PROSTAD"

# Add weekday
for df in (pre_df, pros_df):
    df["weekday"] = df["referral_date"].dt.weekday

# Weekday-only referrals
pre_weekdays = pre_df[pre_df["weekday"] < 5].copy()
pros_weekdays = pros_df[pros_df["weekday"] < 5].copy()

def daily_referral_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("referral_date")
        .size()
        .reset_index(name="n_referrals")
        .sort_values("referral_date")
        .reset_index(drop=True)
    )

def summarise_lambda(df: pd.DataFrame, label: str) -> dict:
    daily = daily_referral_counts(df)
    return {
        "pathway": label,
        "n_referrals": int(len(df)),
        "n_weekdays_with_referrals": int(len(daily)),
        "lam_per_workday_observed_days": float(daily["n_referrals"].mean()) if len(daily) else float("nan"),
        "median_referrals_per_day": float(daily["n_referrals"].median()) if len(daily) else float("nan"),
        "min_date": df["referral_date"].min(),
        "max_date": df["referral_date"].max(),
    }

def weekday_specific_lambda(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = (
        df.groupby("weekday")
        .size()
        .reset_index(name="n_referrals")
    )

    # number of unique calendar dates contributing to each weekday
    weekday_days = (
        df[["referral_date", "weekday"]]
        .drop_duplicates()
        .groupby("weekday")
        .size()
        .reset_index(name="n_days")
    )

    out = out.merge(weekday_days, on="weekday", how="left")
    out["lambda_for_weekday"] = out["n_referrals"] / out["n_days"]
    out["pathway"] = label
    return out[["pathway", "weekday", "n_referrals", "n_days", "lambda_for_weekday"]]

# Summaries
pre_summary = summarise_lambda(pre_weekdays, "BASELINE")
pros_summary = summarise_lambda(pros_weekdays, "PROSTAD")

summary_df = pd.DataFrame([pre_summary, pros_summary])

pre_weekday_lambda = weekday_specific_lambda(pre_weekdays, "BASELINE")
pros_weekday_lambda = weekday_specific_lambda(pros_weekdays, "PROSTAD")
weekday_lambda_df = pd.concat([pre_weekday_lambda, pros_weekday_lambda], ignore_index=True)

# Simple historical proportions
combined_df = pd.concat([pre_weekdays, pros_weekdays], ignore_index=True)
pathway_counts = combined_df["pathway"].value_counts(dropna=False)
p_prostad_historical = pathway_counts.get("PROSTAD", 0) / len(combined_df)
p_baseline_historical = pathway_counts.get("BASELINE", 0) / len(combined_df)

print("\n=== Arrival summaries ===")
print(summary_df.to_string(index=False))

print("\n=== Weekday-specific lambda ===")
print(weekday_lambda_df.to_string(index=False))

print("\n=== Historical proportions ===")
print("p_baseline_historical =", p_baseline_historical)
print("p_prostad_historical  =", p_prostad_historical)

print("\n=== Date ranges ===")
print("PRE: ", pre_weekdays["referral_date"].min(), "to", pre_weekdays["referral_date"].max())
print("PROS:", pros_weekdays["referral_date"].min(), "to", pros_weekdays["referral_date"].max())