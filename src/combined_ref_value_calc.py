import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

pre_df = pd.read_csv(DATA_DIR / "pre_refferal.csv")
prostad_df = pd.read_csv(DATA_DIR / "pros_ref.csv")

# Ensure dates parsed correctly
pre_df["referral_date"] = pd.to_datetime(
    pre_df["Date of referral to pathway"],
    dayfirst=True,
    errors="coerce",
)

prostad_df["referral_date"] = pd.to_datetime(
    prostad_df["Date of referral to pathway"],
    format="%m/%d/%y",
    errors="coerce",
)

# Add pathway label
pre_df["pathway"] = "BASELINE"
prostad_df["pathway"] = "PROSTAD"


def daily_counts_including_zero_weekdays(df: pd.DataFrame) -> pd.Series:
    """
    Return weekday referral counts including zero-referral weekdays.
    """
    dates = df["referral_date"].dropna().dt.normalize()

    # Keep referrals that happened on weekdays
    weekday_dates = dates[dates.dt.weekday < 5]

    if weekday_dates.empty:
        return pd.Series(dtype=int)

    # Full weekday range across the observation window
    all_weekdays = pd.date_range(
        start=weekday_dates.min(),
        end=weekday_dates.max(),
        freq="B",
    )

    # Count observed referrals per weekday
    daily_counts = weekday_dates.value_counts()
    daily_counts.index = pd.to_datetime(daily_counts.index).normalize()

    # Add missing weekdays as zero-referral days
    daily_counts = daily_counts.reindex(all_weekdays, fill_value=0).sort_index()

    return daily_counts


# ------------------------------------------------------------------
# Individual lambda estimates
# ------------------------------------------------------------------

pre_daily = daily_counts_including_zero_weekdays(pre_df)
pros_daily = daily_counts_including_zero_weekdays(prostad_df)

lam_pre = pre_daily.mean()
lam_pros = pros_daily.mean()

print("PRE:", pre_df["referral_date"].min(), pre_df["referral_date"].max())
print("PROS:", prostad_df["referral_date"].min(), prostad_df["referral_date"].max())

print("\nPre-PROSTAD")
print("Total weekday referrals:", pre_daily.sum())
print("Number of weekdays:", len(pre_daily))
print("lam_pre =", lam_pre)

print("\nPROSTAD")
print("Total weekday referrals:", pros_daily.sum())
print("Number of weekdays:", len(pros_daily))
print("lam_pros =", lam_pros)


# ------------------------------------------------------------------
# Combined data
# ------------------------------------------------------------------

df = pd.concat([pre_df, prostad_df], ignore_index=True)

counts = df["pathway"].value_counts()
p_prostad = counts["PROSTAD"] / len(df)

combined_daily = daily_counts_including_zero_weekdays(df)
lam_combined = combined_daily.mean()

print("\nCombined")
print("p_prostad =", p_prostad)
print("Total weekday referrals:", combined_daily.sum())
print("Number of weekdays:", len(combined_daily))
print("lam_combined =", lam_combined)

print("\nCheck")
print("lam_pre + lam_pros =", lam_pre + lam_pros)