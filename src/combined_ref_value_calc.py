import pandas as pd
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

pre_df = pd.read_csv(DATA_DIR /"pre_refferal.csv")
prostad_df = pd.read_csv(DATA_DIR / "pros_ref.csv")

# Ensure dates parsed correctly
pre_df["referral_date"] = pd.to_datetime(pre_df["Date of referral to pathway"], dayfirst=True)
prostad_df["referral_date"] = pd.to_datetime(prostad_df["Date of referral to pathway"], format="%m/%d/%y")

# Add pathway label
pre_df["pathway"] = "BASELINE"
prostad_df["pathway"] = "PROSTAD"

# Combine
df = pd.concat([pre_df, prostad_df], ignore_index=True)


counts = df["pathway"].value_counts()

p_prostad = counts["PROSTAD"] / len(df)

print("p_prostad =", p_prostad)


# Add weekday
df["weekday"] = df["referral_date"].dt.weekday

# Keep weekdays only (Mon=0, Fri=4)
weekday_df = df[df["weekday"] < 5]

# Count referrals per day
daily_counts = (
    weekday_df.groupby("referral_date")
    .size()
    .reset_index(name="n_referrals")
)

lam_per_workday = daily_counts["n_referrals"].mean()

print("lam_per_workday =", lam_per_workday)


pre_daily = (
    pre_df.assign(referral_date=pd.to_datetime(pre_df["referral_date"]))
    .query("referral_date.dt.weekday < 5", engine="python")
    .groupby("referral_date")
    .size()
)

pros_daily = (
    prostad_df.assign(referral_date=pd.to_datetime(prostad_df["referral_date"]))
    .query("referral_date.dt.weekday < 5", engine="python")
    .groupby("referral_date")
    .size()
)

lam_pre = pre_daily.mean()
lam_pros = pros_daily.mean()
lam_combined = pd.concat([pre_df, prostad_df], ignore_index=True)

combined_daily = (
    pd.concat([pre_df, prostad_df], ignore_index=True)
    .assign(referral_date=lambda d: pd.to_datetime(d["referral_date"]))
    .query("referral_date.dt.weekday < 5", engine="python")
    .groupby("referral_date")
    .size()
)

print("lam_pre =", lam_pre)
print("lam_pros =", lam_pros)
print("lam_pre + lam_pros =", lam_pre + lam_pros)
print("lam_combined =", combined_daily.mean())

print("PRE:", pre_df["referral_date"].min(), pre_df["referral_date"].max())
print("PROS:", prostad_df["referral_date"].min(), prostad_df["referral_date"].max())


