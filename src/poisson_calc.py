import pandas as pd
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"



def estimate_lambda_per_workday(df, date_column):

    # convert to datetime
    dates = pd.to_datetime(df[date_column]).dt.normalize()

    # keep weekdays only
    weekday_dates = dates[dates.dt.weekday < 5]

    # count total referrals
    total_referrals = len(weekday_dates)

    # count referrals per date
    daily_counts = weekday_dates.value_counts()

    # create full weekday range (includes zero-referral days)
    all_weekdays = pd.date_range(
        start=weekday_dates.min(),
        end=weekday_dates.max(),
        freq="B"  # business days
    )

     # count number of weekdays
    n_weekdays = len(all_weekdays)

    # calculate lambda
    lam = total_referrals / n_weekdays


    # fill missing days with 0
    #daily_counts = daily_counts.reindex(all_weekdays, fill_value=0)
    #print(daily_counts.describe())

    # calculate lambda
    #lam = daily_counts.mean()
    print("Total referrals:", total_referrals)
    print("Number of weekdays:", n_weekdays)

    #return lam


#Computing the Poisson probability mass function (PMF) for values 0,1,2,…,k 0,1,2,…,k without using factorials. It uses a stable recurrence formula, which is numerically safer.
def poisson_pmf_0_to_k(lam: float, k: int) -> np.ndarray:
    """Stable Poisson pmf for 0..k via recurrence."""
    p = np.zeros(k + 1, dtype=float)
    p[0] = np.exp(-lam)
    for i in range(k):
        p[i + 1] = p[i] * lam / (i + 1)
    return p


def make_poisson_validation_plot(
    df,
    date_column: str,
    outpath: str = "poisson_arrival_validation.png",
    k_max: int = 10,
):
    """
    One-plot validation:
      - histogram of observed weekday referrals/day
      - overlay expected Poisson(lam) counts/day (scaled by number of weekdays)
    """
    # --- observed daily weekday counts (including zeros) ---
    dates = pd.to_datetime(df[date_column]).dt.normalize()
    weekday_dates = dates[dates.dt.weekday < 5]

    all_weekdays = pd.date_range(
        start=weekday_dates.min(),
        end=weekday_dates.max(),
        freq="B"
    )

    # counts for days with referrals
    daily_counts = weekday_dates.value_counts()
    daily_counts.index = pd.to_datetime(daily_counts.index).normalize()

    # include zero-referral weekdays
    obs_counts = daily_counts.reindex(all_weekdays, fill_value=0).sort_index()
    lam = obs_counts.sum() / len(obs_counts)

    # --- expected Poisson counts (in "number of days") ---
    # We make bins 0..k_max-1 and a tail bin >=k_max to avoid tiny expected values
    pmf = poisson_pmf_0_to_k(lam, k_max - 1)
    expected = np.zeros(k_max + 1, dtype=float)
    expected[:k_max] = len(obs_counts) * pmf
    expected[k_max] = len(obs_counts) * (1.0 - pmf.sum())  # tail

    # --- observed binned similarly ---
    obs_binned = np.zeros(k_max + 1, dtype=float)
    for x in obs_counts.values:
        if x >= k_max:
            obs_binned[k_max] += 1
        else:
            obs_binned[int(x)] += 1

    # --- plot ---
    x = np.arange(k_max + 1)
    x_labels = [str(i) for i in range(k_max)] + [f"{k_max}+"]

    plt.figure()
    plt.bar(x, obs_binned, alpha=0.75, label="Observed (weekday daily counts)")
    plt.plot(x, expected, marker="o", linewidth=2, label=f"Expected Poisson(λ={lam:.2f})")

    plt.xticks(x, x_labels)
    plt.xlabel("Referrals per weekday (binned)")
    plt.ylabel("Number of weekdays")
    plt.title("Poisson arrival model validation (weekdays only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return lam


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "pre_refferal.csv")
    lam = make_poisson_validation_plot(
        df,
        date_column="Date of referral to pathway",  
        outpath="poisson_arrival_validation.png",
        k_max=10,
    )
    print(f"Saved poisson_arrival_validation.png (lambda={lam:.3f})")







    