from datetime import date
from des_engine import DayLoopConfig, run_day_loop
from single_walk_mdt_day import trace_one_patient_mdtday 

cfg = DayLoopConfig(
    start_date=date(2024, 1, 1),
    n_days=365,
    lam_per_workday=0.586, #calculated in poisson_calc.py
    seed=42,
)

results = run_day_loop(cfg, trace_one_patient_mdtday)
print(results["summary_stats"])


#validation plot
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "date": results["daily_referrals"].keys(),
    "referrals": results["daily_referrals"].values()
})

df["weekday"] = pd.to_datetime(df["date"]).dt.weekday

weekday_df = df[df["weekday"] < 5]

plt.hist(weekday_df["referrals"], bins=range(0,6))
plt.xlabel("Referrals per weekday")
plt.ylabel("Frequency")
plt.title("Simulated weekday referral counts")
plt.show()