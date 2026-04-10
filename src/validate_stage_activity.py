import numpy as np
import pandas as pd
from datetime import date

from scenarios import build_scenario_config
from validate_scenarios import run_scenario



def summarise_stage_demand(result):
    rows = []

    for stage_name, metrics in result["stage_activity"].items():
        arrivals = metrics["daily_arrivals"]
        in_stage = metrics["daily_in_stage"]
        completed = metrics["daily_completed"]

        rows.append({
            "stage": stage_name,
            "total_arrivals": sum(arrivals.values()),
            "mean_daily_arrivals": np.mean(list(arrivals.values())) if arrivals else 0,
            "peak_daily_arrivals": max(arrivals.values()) if arrivals else 0,
            "mean_in_stage": np.mean(list(in_stage.values())) if in_stage else 0,
            "peak_in_stage": max(in_stage.values()) if in_stage else 0,
            "total_completed": sum(completed.values()),
            "completion_ratio" : sum(completed.values()) / sum(arrivals.values()),
        })

    return pd.DataFrame(rows)

mc_res = run_scenario("ALL_MC_BASELINE", start_date=date(2026, 1, 5),
        n_days=365,
        lam_per_workday=0.586,
        seed=123,)
prostad_res = run_scenario(name="PROSTAD",
        start_date=date(2026, 1, 5),
        n_days=365,
        lam_per_workday=0.586,
        seed=123,)

mc_demand = summarise_stage_demand(mc_res).rename(columns=lambda c: f"mc_{c}" if c != "stage" else c)
prostad_demand = summarise_stage_demand(prostad_res).rename(columns=lambda c: f"prostad_{c}" if c != "stage" else c)

comparison = mc_demand.merge(prostad_demand, on="stage")
comparison["arrival_change"] = comparison["prostad_total_arrivals"] - comparison["mc_total_arrivals"]
comparison["in_stage_change"] = comparison["prostad_mean_in_stage"] - comparison["mc_mean_in_stage"]
comparison["peak_in_stage_change"] = comparison["prostad_peak_in_stage"] - comparison["mc_peak_in_stage"]
comparison["completion_change"] = comparison["prostad_completion_ratio"] - comparison["mc_completion_ratio"]

#print(mc_demand)
#print(prostad_demand)
#print(comparison)



# % changes (handle divide-by-zero safely)
def safe_pct_change(new, old):
    return np.where(old != 0, (new - old) / old * 100, np.nan)

comparison["arrival_pct_change"] = safe_pct_change(
    comparison["prostad_total_arrivals"],
    comparison["mc_total_arrivals"],
)

comparison["in_stage_pct_change"] = safe_pct_change(
    comparison["prostad_mean_in_stage"],
    comparison["mc_mean_in_stage"],
)

comparison["completion_pct_change"] = safe_pct_change(
    comparison["prostad_completion_ratio"],
    comparison["mc_completion_ratio"],
)

def classify_stage(row):
    # Strong bottleneck: demand ↑ and queue ↑ and not clearing
    if (
        row["arrival_change"] > 5 and
        row["in_stage_change"] > 1 and
        row["prostad_completion_ratio"] < 0.95
    ):
        return "🔴 BOTTLENECK"

    # Pressure increasing but still coping
    if row["in_stage_change"] > 0.5:
        return "🟠 PRESSURE ↑"

    # Improved flow (queues reduced)
    if row["in_stage_change"] < -0.5:
        return "🟢 IMPROVED"

    # Fast / near-instant stage
    if row["prostad_mean_in_stage"] < 0.5:
        return "⚡ FAST TRACK"

    return "⚪ STABLE"


comparison["stage_flag"] = comparison.apply(classify_stage, axis=1)

summary_cols = [
    "stage",

    "mc_total_arrivals",
    "prostad_total_arrivals",
    "arrival_change",
    "arrival_pct_change",

    "mc_mean_in_stage",
    "prostad_mean_in_stage",
    "in_stage_change",
    "in_stage_pct_change",

    "mc_completion_ratio",
    "prostad_completion_ratio",
    "completion_change",

    "stage_flag",
]

summary_table = comparison[summary_cols].copy()
summary_table = summary_table.round(3)
print("\n=== MC vs PROSTAD Demand & Bottleneck Summary ===")
print(summary_table.to_string(index=False))

import matplotlib.pyplot as plt

# -----------------------------------------
# Plot: mean number in stage (MC vs PROSTAD)
# -----------------------------------------
plot_df = comparison.copy()

# Optional: cleaner stage labels for display
stage_labels = {
    "ref_to_mri": "Referral → MRI",
    "mri_to_report": "MRI → Report",
    "report_to_biopmdt": "Report → Biopsy MDT",
    "biopmdt_to_biopsy": "Biopsy MDT → Biopsy",
    "biopsy_to_pathrep": "Biopsy → Path Report",
    "pathrep_to_treatmdt": "Path Report → Treat MDT",
    "treatmdt_to_outpat": "Treat MDT → Outpatient",
}

plot_df["stage_label"] = plot_df["stage"].map(stage_labels).fillna(plot_df["stage"])

x = np.arange(len(plot_df))
width = 0.38

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, plot_df["mc_mean_in_stage"], width, label="ALL_MC_BASELINE")
plt.bar(x + width/2, plot_df["prostad_mean_in_stage"], width, label="PROSTAD")

plt.xticks(x, plot_df["stage_label"], rotation=30, ha="right")
plt.ylabel("Mean number in stage")
plt.title("Bottleneck shift: mean number in stage by scenario")
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("outputs/bottleneck_shift_mean_in_stage.png", dpi=300, bbox_inches="tight")

# -----------------------------------------
# Plot: change in mean number in stage
# -----------------------------------------
delta_df = comparison.copy()
delta_df["stage_label"] = delta_df["stage"].map(stage_labels).fillna(delta_df["stage"])

plt.figure(figsize=(12, 6))
plt.bar(delta_df["stage_label"], delta_df["in_stage_change"])
plt.axhline(0, linewidth=1)
plt.xticks(rotation=30, ha="right")
plt.ylabel("PROSTAD - MC mean in stage")
plt.title("Change in stage occupancy under PROSTAD")
plt.tight_layout()
plt.show()

plt.savefig("outputs/bottleneck_shift_delta_in_stage.png", dpi=300, bbox_inches="tight")