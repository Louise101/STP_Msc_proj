from __future__ import annotations

from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from combined_des_engine import CombinedEngineConfig, run_day_loop_combined_engine

def build_sim_weekly_biopsy_df(event_log: pd.DataFrame, sim_start_date) -> pd.DataFrame:
    df = event_log.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    biopsy_df = df[df["event"] == "biopsy_done"].copy()

    sim_start = pd.to_datetime(sim_start_date)
    biopsy_df["days_since_start"] = (biopsy_df["date"] - sim_start).dt.days
    biopsy_df = biopsy_df[biopsy_df["days_since_start"] >= 0].copy()

    biopsy_df["week_index"] = (biopsy_df["days_since_start"] // 7).astype(int)

    weekly = (
        biopsy_df.groupby("week_index")
        .size()
        .reset_index(name="sim_biopsies")
        .sort_values("week_index")
    )
    return weekly


def build_real_weekly_biopsy_df(pre_file: str, pros_file: str) -> pd.DataFrame:
    pre_df = pd.read_csv(pre_file)
    pros_df = pd.read_csv(pros_file)

    pre_df["biopsy_date"] = pd.to_datetime(
        pre_df["Date of Biopsy"],
        dayfirst=True,
        errors="coerce",
    )
    pros_df["biopsy_date"] = pd.to_datetime(
        pros_df["Date of biopsy"],
        format="%m/%d/%y",
        errors="coerce",
    )

    pre_df = pre_df[["biopsy_date"]].dropna().copy()
    pros_df = pros_df[["biopsy_date"]].dropna().copy()

    real_df = pd.concat([pre_df, pros_df], ignore_index=True).sort_values("biopsy_date")

    real_start = real_df["biopsy_date"].min()
    real_df["days_since_start"] = (real_df["biopsy_date"] - real_start).dt.days
    real_df["week_index"] = (real_df["days_since_start"] // 7).astype(int)

    weekly = (
        real_df.groupby("week_index")
        .size()
        .reset_index(name="real_biopsies")
        .sort_values("week_index")
    )
    return weekly


def compare_weekly_biopsy(real_weekly: pd.DataFrame, sim_weekly: pd.DataFrame) -> pd.DataFrame:
    df_compare = pd.merge(real_weekly, sim_weekly, on="week_index", how="outer").sort_values("week_index")
    df_compare["real_biopsies"] = df_compare["real_biopsies"].fillna(0).astype(int)
    df_compare["sim_biopsies"] = df_compare["sim_biopsies"].fillna(0).astype(int)
    return df_compare


def plot_weekly_biopsy_comparison(df_compare: pd.DataFrame, out_file: str | None = None) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df_compare["week_index"], df_compare["real_biopsies"], label="Real", marker="o")
    plt.plot(df_compare["week_index"], df_compare["sim_biopsies"], label="Simulated", marker="o")
    plt.xlabel("Week index from start")
    plt.ylabel("Biopsies per week")
    plt.title("Weekly biopsy arrivals: Real vs Simulated")
    plt.legend()
    plt.tight_layout()

    if out_file is not None:
        plt.savefig(out_file, dpi=300, bbox_inches="tight")

    plt.show()


def print_weekly_biopsy_summary(df_compare: pd.DataFrame) -> None:
    print("\n=== Weekly biopsy stats: REAL ===")
    print(df_compare["real_biopsies"].describe())

    print("\n=== Weekly biopsy stats: SIMULATED ===")
    print(df_compare["sim_biopsies"].describe())

    corr = df_compare["real_biopsies"].corr(df_compare["sim_biopsies"])
    print("\nCorrelation:", corr)

def main():
    cfg = CombinedEngineConfig(
        start_date=date(2026, 1, 5),
        n_days=365,
        lam_per_workday=1.7528735632183907, #calculated in combined_ref_value_calc.py
        p_prostad=0.5098039215686274, #calculated in combined_ref_value_calc.py
        mri_capacity_by_weekday_prostad={1: 4},
        seed=1234,
        scenario_name="COMBINED_MIXED_STREAM",
    )

    print("p_prostad =", cfg.p_prostad)

    result = run_day_loop_combined_engine(cfg)

    print("Total completed:", result["summary_stats"]["total_patients_completed"])
    print("Total referrals:", sum(result["daily_referrals"].values()))

    result["event_log"].to_csv("outputs/combined_event_log.csv", index=False)

    all_pts = result["all_patients_objects"]
    n_total = len(all_pts)
    n_pros = sum(1 for p in all_pts if p.data.get("pathway_type") == "PROSTAD")
    n_base = sum(1 for p in all_pts if p.data.get("pathway_type") == "BASELINE")

    print("Total patients:", n_total)
    print("PROSTAD patients:", n_pros, n_pros / n_total if n_total else 0)
    print("BASELINE patients:", n_base, n_base / n_total if n_total else 0)

        # --------------------------------------------------
    # Weekly biopsy validation using week index
    # --------------------------------------------------
    sim_weekly = build_sim_weekly_biopsy_df(
        result["event_log"],
        sim_start_date=cfg.start_date,
    )

    real_weekly = build_real_weekly_biopsy_df(
        pre_file="data/pre_biopmdt_to_biop.csv",
        pros_file="data/pros_mriclin_to_biop.csv",
    )

    df_compare = compare_weekly_biopsy(real_weekly, sim_weekly)

    print("\n=== Weekly biopsy comparison table ===")
    print(df_compare.head(20).to_string(index=False))

    print_weekly_biopsy_summary(df_compare)

    df_compare.to_csv("outputs/weekly_biopsy_real_vs_sim_week_index.csv", index=False)

    plot_weekly_biopsy_comparison(
        df_compare,
        out_file="outputs/weekly_biopsy_real_vs_sim_week_index.png",
    )

    sim_df = result["event_log"]

    n_mdt = (sim_df["event"] == "MDT_occured").sum()
    n_biopsy = (sim_df["event"] == "biopsy_done").sum()

    print("Biopsy rate from MDT:", n_biopsy / n_mdt)




if __name__ == "__main__":
    main()