from pathlib import Path
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from des_engine import EngineConfig, run_day_loop_with_stage_engine, WAIT_MODE_DES, WAIT_MODE_MC
from single_walk_mdt_day import trace_one_patient_mdtday


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Real data helpers
# -----------------------------
def load_real_biopsy_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "pre_biopmdt_to_biop.csv")

    df["biopmdt_date"] = pd.to_datetime(df["Date of Prostate MRI MDT"], dayfirst=True)
    df["biopsy_date"] = pd.to_datetime(df["Date of Biopsy"], dayfirst=True)

    df = df.dropna(subset=["biopmdt_date", "biopsy_date"]).copy()
    return df


def build_weekly_real(real_df: pd.DataFrame) -> pd.DataFrame:
    real_df = real_df.copy()

    real_df["biopmdt_week"] = real_df["biopmdt_date"].dt.to_period("W").apply(lambda r: r.start_time)
    real_df["biopsy_week"] = real_df["biopsy_date"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly_demand = real_df.groupby("biopmdt_week").size()
    weekly_completed = real_df.groupby("biopsy_week").size()

    weekly = pd.DataFrame({
        "demand": weekly_demand,
        "completed": weekly_completed,
    }).fillna(0)

    weekly = weekly.reset_index().rename(columns={"index": "week"})
    weekly["week"] = pd.to_datetime(weekly["week"])

    # true waiting count at each week start
    results = []
    for week in weekly["week"]:
        waiting = real_df[
            (real_df["biopmdt_date"] <= week) &
            (real_df["biopsy_date"] > week)
        ].shape[0]

        results.append({
            "week": week,
            "waiting": waiting,
        })

    waiting_df = pd.DataFrame(results)

    weekly = weekly.merge(waiting_df, on="week", how="left")
    weekly = weekly.sort_values("week").reset_index(drop=True)

    return weekly


# -----------------------------
# Simulation helpers
# -----------------------------
def build_weekly_from_sim(sim_output: dict) -> pd.DataFrame:
    records = []

    for events, _ in sim_output["patient_results"]:
        biopsy_date = None
        for e in events:
            if e["event"] == "biopsy_done":
                biopsy_date = pd.to_datetime(e["date"])
                break

        if biopsy_date is not None:
            records.append({"biopsy_date": biopsy_date})

    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(columns=["week", "completed"])

    df["week"] = df["biopsy_date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = df.groupby("week").size().rename("completed").reset_index()
    weekly["week"] = pd.to_datetime(weekly["week"])

    return weekly


def build_weekly_queue_from_sim(sim_output: dict) -> pd.DataFrame:
    daily_arrivals = pd.Series(sim_output["stage_activity"]["Biopsy"]["daily_arrivals"]).sort_index()
    daily_completed = pd.Series(sim_output["resources"]["Biopsy"]["daily_started"]).sort_index()
    daily_backlog = pd.Series(sim_output["stage_activity"]["Biopsy"]["daily_backlog"]).sort_index()

    daily_arrivals.index = pd.to_datetime(daily_arrivals.index)
    daily_completed.index = pd.to_datetime(daily_completed.index)
    daily_backlog.index = pd.to_datetime(daily_backlog.index)

    weekly = pd.DataFrame({
        "demand": daily_arrivals.resample("W-MON").sum(),
        "completed": daily_completed.resample("W-MON").sum(),
        "mean_backlog": daily_backlog.resample("W-MON").mean(),
        "end_backlog": daily_backlog.resample("W-MON").last(),
    }).fillna(0)

    weekly = weekly.reset_index().rename(columns={"index": "week"})
    return weekly


def extract_sim_waits(sim_output: dict) -> pd.Series:
    waits = []

    for events, _ in sim_output["patient_results"]:
        mdt_date = None
        biopsy_date = None

        for e in events:
            if e["event"] == "MDT_occured":
                mdt_date = e["date"]
            elif e["event"] == "biopsy_done":
                biopsy_date = e["date"]

        if mdt_date is not None and biopsy_date is not None:
            waits.append((biopsy_date - mdt_date).days)

    return pd.Series(waits, dtype="float")


# -----------------------------
# 1. Run simulation
# -----------------------------
cfg = EngineConfig(
    start_date=date(2024, 1, 1),
    n_days=365,
    lam_per_workday=0.586,
    mri_capacity_by_weekday={2: 4},
    biopsy_capacity_by_weekday={3: 1, 4: 1},
    seed=42,
    wait_time_mode={
        "ref_to_mri": WAIT_MODE_MC,
        "mri_to_report": WAIT_MODE_MC,
        "report_to_biopmdt": WAIT_MODE_MC,
        "biopmdt_to_biopsy": WAIT_MODE_DES,
    },
)

sim_output = run_day_loop_with_stage_engine(cfg, trace_one_patient_mdtday)


# -----------------------------
# 2. Build weekly data
# -----------------------------
real_df = load_real_biopsy_data()
weekly_real = build_weekly_real(real_df)

weekly_sim = build_weekly_from_sim(sim_output)
print("\n=== Weekly simulated completed summary ===")
print(weekly_sim["completed"].describe())


# -----------------------------
# 3. Test 1: peaks
# -----------------------------
print("\n=== Weekly completed value counts ===")
print(weekly_sim["completed"].value_counts().sort_index())


# -----------------------------
# 4. Test 2: queue dynamics
# -----------------------------
weekly_queue = build_weekly_queue_from_sim(sim_output)

print("\n=== Weekly simulation queue data ===")
print(weekly_queue.head(20))

plt.figure()
plt.plot(weekly_queue["week"], weekly_queue["end_backlog"])
plt.xticks(rotation=45)
plt.title("Simulated biopsy backlog (end of week)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sim_biopsy_backlog.png", dpi=300)
plt.close()


# -----------------------------
# 5. Test 3: distribution comparison
# -----------------------------
sim_waits = extract_sim_waits(sim_output)
real_waits = (real_df["biopsy_date"] - real_df["biopmdt_date"]).dt.days

print("\n=== Wait summaries ===")
print("REAL:")
print(real_waits.describe())
print("\nSIM:")
print(sim_waits.describe())

ks_stat, ks_p = ks_2samp(real_waits, sim_waits)

print("\n=== KS test ===")
print("KS stat:", ks_stat)
print("KS p:", ks_p)


# -----------------------------
# 6. Test 4: correlations
# -----------------------------
weekly_real["prev_waiting"] = weekly_real["waiting"].shift(1)
weekly_queue["prev_waiting"] = weekly_queue["end_backlog"].shift(1)

print("\n=== REAL correlations ===")
print(weekly_real[["demand", "completed", "prev_waiting"]].corr())
print("Completed vs demand:", weekly_real["completed"].corr(weekly_real["demand"]))
print("Completed vs previous waiting:", weekly_real["completed"].corr(weekly_real["prev_waiting"]))

print("\n=== SIM correlations ===")
print(weekly_queue[["demand", "completed", "prev_waiting"]].corr())
print("Completed vs demand:", weekly_queue["completed"].corr(weekly_queue["demand"]))
print("Completed vs previous queue:", weekly_queue["completed"].corr(weekly_queue["prev_waiting"]))

plt.figure()
plt.scatter(weekly_queue["prev_waiting"], weekly_queue["completed"])
plt.xlabel("Previous backlog")
plt.ylabel("Completed")
plt.title("Simulated capacity response to backlog")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sim_backlog_vs_completed.png", dpi=300)
plt.close()

plt.figure()
plt.hist(real_waits, bins=15, alpha=0.6, label="Real")
plt.hist(sim_waits, bins=15, alpha=0.6, label="Sim")
plt.xlabel("MDT to biopsy wait (days)")
plt.ylabel("Count")
plt.title("Real vs simulated MDT-to-biopsy waits")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "real_vs_sim_biopsy_waits_hist.png", dpi=300)
plt.close()

#summary table
summary_table = pd.DataFrame([
    {
        "metric": "real_mean_wait",
        "value": float(real_waits.mean()),
    },
    {
        "metric": "sim_mean_wait",
        "value": float(sim_waits.mean()),
    },
    {
        "metric": "real_median_wait",
        "value": float(real_waits.median()),
    },
    {
        "metric": "sim_median_wait",
        "value": float(sim_waits.median()),
    },
    {
        "metric": "ks_stat",
        "value": float(ks_stat),
    },
    {
        "metric": "ks_p",
        "value": float(ks_p),
    },
    {
        "metric": "real_corr_completed_vs_prev_waiting",
        "value": float(weekly_real["completed"].corr(weekly_real["prev_waiting"])),
    },
    {
        "metric": "sim_corr_completed_vs_prev_queue",
        "value": float(weekly_queue["completed"].corr(weekly_queue["prev_waiting"])),
    },
    {
        "metric": "real_corr_completed_vs_demand",
        "value": float(weekly_real["completed"].corr(weekly_real["demand"])),
    },
    {
        "metric": "sim_corr_completed_vs_demand",
        "value": float(weekly_queue["completed"].corr(weekly_queue["demand"])),
    },
    {
        "metric": "final_backlog_biopsy",
        "value": float(sim_output["summary_stats"]["final_backlog_by_resource"]["Biopsy"]),
    },
])


# -----------------------------
# 7. Simple DES vs observed comparison
# -----------------------------
mean_des = sim_waits.mean()
mean_mc = real_waits.mean()

print("\n=== Mean wait comparison ===")
print("Mean wait DES:", mean_des)
print("Mean wait observed:", mean_mc)

print("\n=== Summary stats from sim ===")
print(sim_output["summary_stats"])


# -----------------------------
# 8. Save outputs
# -----------------------------
weekly_sim.to_csv(OUTPUT_DIR / "weekly_sim_completed.csv", index=False)
weekly_queue.to_csv(OUTPUT_DIR / "weekly_sim_queue.csv", index=False)
weekly_real.to_csv(OUTPUT_DIR / "weekly_real_biopsy.csv", index=False)
pd.DataFrame({"sim_waits": sim_waits}).to_csv(OUTPUT_DIR / "sim_mdt_to_biopsy_waits.csv", index=False)
pd.DataFrame({"real_waits": real_waits}).to_csv(OUTPUT_DIR / "real_mdt_to_biopsy_waits.csv", index=False)
summary_table.to_csv(OUTPUT_DIR / "biopsy_validation_summary.csv", index=False)

print("\nSaved outputs to:", OUTPUT_DIR)