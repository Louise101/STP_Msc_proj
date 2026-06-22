# Save as: src/analysis/animate_combined_pathway.py

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from engine.combined_engine import run_day_loop_combined_engine
from engine.scenarios import build_combined_config, generate_daily_referrals


BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "combined_animation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_GIF = OUTPUT_DIR / "combined_pathway_animation_m3.gif"
OUTPUT_MP4 = OUTPUT_DIR / "combined_pathway_animation_m3.mp4"

START_DATE = date(2024, 1, 5)
N_DAYS = 365
LAM_PER_WORKDAY = 1.1010830324909748
SEED = 1000
SCENARIO_NAME = "OBS_MIX"

MAX_PATIENTS_TO_ANIMATE = 400
FPS = 4
RNG_SEED = 1234


EVENT_ORDER = [
    "referral_received",
    "mri_performed",
    "mri_report_ready",
    "MDT_occured",
    "biopsy_done",
    "Path_report_recieved",
    "Treatment_options_MDT_occured",
    "Outpatient_appointment_occured",
]

STAGE_STATES = [
    "queue_ref_to_mri",
    "queue_mri_to_report",
    "queue_report_to_biopmdt",
    "queue_biopmdt_to_biopsy",
    "queue_biopsy_to_pathrep",
    "queue_pathrep_to_treatmdt",
    "queue_treatmdt_to_outpat",
]

EXIT_STATES = [
    "exit_after_biopmdt",
    "exit_after_pathrep",
    "exit_after_outpatient",
]

STATE_LABELS = {
    "queue_ref_to_mri": "Referral → MRI",
    "queue_mri_to_report": "MRI → Report",
    "queue_report_to_biopmdt": "Report → Decision",
    "queue_biopmdt_to_biopsy": "Decision → Biopsy",
    "queue_biopsy_to_pathrep": "Biopsy → Pathology",
    "queue_pathrep_to_treatmdt": "Pathology → Treatment MDT",
    "queue_treatmdt_to_outpat": "Treatment MDT → Outpatient",
    "exit_after_biopmdt": "Exit after decision",
    "exit_after_pathrep": "Exit after pathology",
    "exit_after_outpatient": "Pathway complete",
}

OUTCOME_COLOUR_MAP = {
    "exit_after_outpatient": "tab:blue",
    "exit_after_pathrep": "tab:orange",
    "exit_after_biopmdt": "tab:red",
    "unknown": "tab:gray",
}

EXIT_BOX_FACECOLOURS = {
    "exit_after_biopmdt": "mistyrose",
    "exit_after_pathrep": "moccasin",
    "exit_after_outpatient": "lightblue",
}

PATHWAY_Y_BASE = {
    "BASELINE": 6.0,
    "PROSTAD": 2.2,
}

X_START = 0.6
BOX_W = 2.2
BOX_H = 1.25
X_GAP = 1.45

STATE_LAYOUT: dict[tuple[str, str], tuple[float, float, float, float]] = {}

for pathway_type, y in PATHWAY_Y_BASE.items():
    for i, state in enumerate(STAGE_STATES):
        STATE_LAYOUT[(pathway_type, state)] = (
            X_START + i * (BOX_W + X_GAP),
            y,
            BOX_W,
            BOX_H,
        )

    STATE_LAYOUT[(pathway_type, "exit_after_biopmdt")] = (
        X_START + 2 * (BOX_W + X_GAP),
        y - 1.65,
        BOX_W,
        BOX_H,
    )
    STATE_LAYOUT[(pathway_type, "exit_after_pathrep")] = (
        X_START + 4 * (BOX_W + X_GAP),
        y - 1.65,
        BOX_W,
        BOX_H,
    )
    STATE_LAYOUT[(pathway_type, "exit_after_outpatient")] = (
        X_START + 6 * (BOX_W + X_GAP),
        y - 1.65,
        BOX_W,
        BOX_H,
    )


def run_combined_model() -> dict:
    referral_schedule = generate_daily_referrals(
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=SEED,
    )

    cfg = build_combined_config(
        SCENARIO_NAME,
        start_date=START_DATE,
        n_days=N_DAYS,
        lam_per_workday=LAM_PER_WORKDAY,
        seed=SEED,
    )

    return run_day_loop_combined_engine(
        cfg,
        daily_referrals_override=referral_schedule,
    )


def patient_objects_to_wide(result: dict) -> pd.DataFrame:
    rows = []

    for patient in result.get("all_patients_objects", []):
        pathway_type = patient.data.get(
            "pathway_type",
            getattr(patient, "pathway_type", "UNKNOWN"),
        )

        row = {
            "patient_id": patient.patient_id,
            "pathway_type": pathway_type,
            "exit_reason": getattr(patient, "exit_reason", None),
        }

        for event in patient.events:
            event_name = event.get("event")
            event_date = event.get("date")

            if event_name in EVENT_ORDER and event_date is not None:
                event_date = pd.Timestamp(event_date).normalize()

                if event_name not in row:
                    row[event_name] = event_date
                else:
                    row[event_name] = min(row[event_name], event_date)

        rows.append(row)

    wide = pd.DataFrame(rows)

    for event_name in EVENT_ORDER:
        if event_name not in wide.columns:
            wide[event_name] = pd.NaT

    return wide


def sample_patients(wide: pd.DataFrame, max_patients: int, seed: int) -> pd.DataFrame:
    if len(wide) <= max_patients:
        return wide.copy()

    rng = np.random.default_rng(seed)
    keep_ids = rng.choice(
        wide["patient_id"].to_numpy(),
        size=max_patients,
        replace=False,
    )

    return wide[wide["patient_id"].isin(keep_ids)].copy()


def daterange(start_date: pd.Timestamp, end_date: pd.Timestamp):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def add_interval(
    records: list[dict],
    patient_id: int,
    pathway_type: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    state: str,
) -> None:
    if pd.isna(start_date) or pd.isna(end_date):
        return

    if end_date < start_date:
        return

    for current_date in daterange(start_date, end_date):
        records.append(
            {
                "patient_id": patient_id,
                "pathway_type": pathway_type,
                "date": current_date,
                "state": state,
            }
        )


def build_daily_states(wide: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []

    all_dates = []
    for event_name in EVENT_ORDER:
        all_dates.extend(wide[event_name].dropna().tolist())

    if not all_dates:
        raise ValueError("No event dates found.")

    global_end = max(all_dates) + timedelta(days=8)

    for _, row in wide.iterrows():
        pid = row["patient_id"]
        pathway_type = row["pathway_type"]

        ref = row["referral_received"]
        mri = row["mri_performed"]
        rep = row["mri_report_ready"]
        bmdt = row["MDT_occured"]
        biopsy = row["biopsy_done"]
        pathrep = row["Path_report_recieved"]
        treat = row["Treatment_options_MDT_occured"]
        outpat = row["Outpatient_appointment_occured"]

        if pd.notna(ref):
            if pd.notna(mri):
                add_interval(records, pid, pathway_type, ref, mri - timedelta(days=1), "queue_ref_to_mri")
            else:
                add_interval(records, pid, pathway_type, ref, global_end, "queue_ref_to_mri")
                continue

        if pd.notna(mri):
            if pd.notna(rep):
                add_interval(records, pid, pathway_type, mri + timedelta(days=1), rep - timedelta(days=1), "queue_mri_to_report")
            else:
                add_interval(records, pid, pathway_type, mri + timedelta(days=1), global_end, "queue_mri_to_report")
                continue

        if pd.notna(rep):
            if pd.notna(bmdt):
                add_interval(records, pid, pathway_type, rep + timedelta(days=1), bmdt - timedelta(days=1), "queue_report_to_biopmdt")
            else:
                add_interval(records, pid, pathway_type, rep + timedelta(days=1), global_end, "exit_after_biopmdt")
                continue

        if pd.notna(bmdt) and pd.notna(biopsy):
            add_interval(records, pid, pathway_type, bmdt + timedelta(days=1), biopsy - timedelta(days=1), "queue_biopmdt_to_biopsy")

            if pd.notna(pathrep):
                add_interval(records, pid, pathway_type, biopsy + timedelta(days=1), pathrep - timedelta(days=1), "queue_biopsy_to_pathrep")
            else:
                add_interval(records, pid, pathway_type, biopsy + timedelta(days=1), global_end, "queue_biopsy_to_pathrep")
                continue

            if pd.notna(treat):
                add_interval(records, pid, pathway_type, pathrep + timedelta(days=1), treat - timedelta(days=1), "queue_pathrep_to_treatmdt")

                if pd.notna(outpat):
                    add_interval(records, pid, pathway_type, treat + timedelta(days=1), outpat - timedelta(days=1), "queue_treatmdt_to_outpat")
                    add_interval(records, pid, pathway_type, outpat + timedelta(days=1), global_end, "exit_after_outpatient")
                else:
                    add_interval(records, pid, pathway_type, treat + timedelta(days=1), global_end, "queue_treatmdt_to_outpat")
            else:
                add_interval(records, pid, pathway_type, pathrep + timedelta(days=1), global_end, "exit_after_pathrep")

        elif pd.notna(bmdt) and pd.isna(biopsy):
            add_interval(records, pid, pathway_type, bmdt + timedelta(days=1), global_end, "exit_after_biopmdt")

    daily = pd.DataFrame(records)

    if daily.empty:
        raise ValueError("No daily states created.")

    daily = daily.drop_duplicates(
        subset=["patient_id", "date"],
        keep="last",
    )

    return daily.sort_values(["date", "pathway_type", "patient_id"]).reset_index(drop=True)


def assign_patient_outcome_colours(daily_states: pd.DataFrame) -> dict:
    patient_colour = {}

    for patient_id, grp in daily_states.groupby("patient_id"):
        states = grp.sort_values("date")["state"].tolist()

        final_exit = "unknown"
        for state in reversed(states):
            if state in EXIT_STATES:
                final_exit = state
                break

        patient_colour[patient_id] = OUTCOME_COLOUR_MAP.get(final_exit, "tab:gray")

    return patient_colour


def assign_dot_positions(daily: pd.DataFrame, seed: int = 1234) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positioned_frames = []

    for _, day_df in daily.groupby("date"):
        for (pathway_type, state), state_df in day_df.groupby(["pathway_type", "state"]):
            key = (pathway_type, state)

            if key not in STATE_LAYOUT:
                continue

            x, y, w, h = STATE_LAYOUT[key]
            n = len(state_df)

            cols = max(1, math.ceil(math.sqrt(n)))
            rows = max(1, math.ceil(n / cols))

            x_margin = 0.12
            y_margin = 0.12
            usable_w = max(0.1, w - 2 * x_margin)
            usable_h = max(0.1, h - 2 * y_margin)

            x_step = usable_w / cols
            y_step = usable_h / rows

            state_df = state_df.sort_values("patient_id").copy()
            coords = []

            for i in range(n):
                col = i % cols
                row = i // cols

                px = x + x_margin + (col + 0.5) * x_step
                py = y + y_margin + usable_h - (row + 0.5) * y_step

                px += rng.uniform(-0.03, 0.03)
                py += rng.uniform(-0.03, 0.03)

                coords.append((px, py))

            state_df["x"] = [c[0] for c in coords]
            state_df["y"] = [c[1] for c in coords]

            positioned_frames.append(state_df)

    return pd.concat(positioned_frames, ignore_index=True)


def draw_static_layout(ax) -> None:
    for (pathway_type, state), (x, y, w, h) in STATE_LAYOUT.items():
        facecolor = EXIT_BOX_FACECOLOURS.get(state, "none")

        rect = Rectangle(
            (x, y),
            w,
            h,
            fill=True,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.4,
        )
        ax.add_patch(rect)

        ax.text(
            x + w / 2,
            y + h + 0.08,
            STATE_LABELS.get(state, state),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.text(
        -0.2,
        PATHWAY_Y_BASE["BASELINE"] + BOX_H / 2,
        "STANDARD\nPATHWAY",
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    ax.text(
        -0.2,
        PATHWAY_Y_BASE["PROSTAD"] + BOX_H / 2,
        "PROSTAD\nPATHWAY",
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    for _, y in PATHWAY_Y_BASE.items():
        arrow_y = y + BOX_H / 2

        for i in range(len(STAGE_STATES) - 1):
            x1 = X_START + i * (BOX_W + X_GAP) + BOX_W
            x2 = X_START + (i + 1) * (BOX_W + X_GAP)

            ax.annotate(
                "",
                xy=(x2 - 0.15, arrow_y),
                xytext=(x1 + 0.15, arrow_y),
                arrowprops=dict(arrowstyle="->", linewidth=1.2),
            )

    ax.set_xlim(-1.7, 28)
    ax.set_ylim(0.2, 8.2)
    ax.set_aspect("equal")
    ax.axis("off")


def make_animation(
    positioned_daily: pd.DataFrame,
    save_gif: bool = False,
    save_mp4: bool = True,
) -> None:
    dates = sorted(positioned_daily["date"].dropna().unique())
    patient_colour_map = assign_patient_outcome_colours(positioned_daily)

    fig, ax = plt.subplots(figsize=(20, 7))

    def update(frame_idx):
        ax.clear()
        draw_static_layout(ax)

        current_date = dates[frame_idx]
        frame_df = positioned_daily[positioned_daily["date"] == current_date].copy()

        colours = frame_df["patient_id"].map(patient_colour_map).fillna("tab:gray")

        ax.scatter(
            frame_df["x"],
            frame_df["y"],
            s=22,
            alpha=0.82,
            c=colours,
        )

        ax.text(
            0.01,
            1.0,
            f"Day {frame_idx} | Date: {pd.Timestamp(current_date).date()}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

        baseline_n = (frame_df["pathway_type"] == "BASELINE").sum()
        prostad_n = (frame_df["pathway_type"] == "PROSTAD").sum()

        ax.text(
            0.01,
            0.96,
            f"Visible patients — Standard: {baseline_n} | PROSTAD: {prostad_n}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label="Completed pathway",
                   markerfacecolor="tab:blue", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Exit after pathology",
                   markerfacecolor="tab:orange", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Exit after decision",
                   markerfacecolor="tab:red", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Unknown / unresolved",
                   markerfacecolor="tab:gray", markersize=8),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.62, 1.02),
            ncol=4,
            frameon=True,
            fontsize=9,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(dates),
        interval=1000 / FPS,
        blit=False,
        repeat=False,
    )

    if save_gif:
        print(f"Saving GIF to {OUTPUT_GIF}")
        anim.save(OUTPUT_GIF, writer=PillowWriter(fps=FPS))

    if save_mp4:
        print(f"Saving MP4 to {OUTPUT_MP4}")
        writer = FFMpegWriter(
            fps=FPS,
            metadata={"artist": "Louise Finlayson"},
            bitrate=1800,
        )
        anim.save(
    OUTPUT_MP4,
    writer="ffmpeg",
    fps=FPS,
    dpi=200,
    bitrate=1800
)

    plt.close(fig)

def main() -> None:
    print("Running combined model...")
    result = run_combined_model()

    print("Converting patient objects to milestone table...")
    wide = patient_objects_to_wide(result)

    print("Patients before sampling:", len(wide))
    wide = sample_patients(wide, MAX_PATIENTS_TO_ANIMATE, RNG_SEED)
    print("Patients animated:", len(wide))

    print(wide["pathway_type"].value_counts(dropna=False))

    print("Building daily states...")
    daily = build_daily_states(wide)
    daily.to_csv(OUTPUT_DIR / "combined_animation_daily_states.csv", index=False)

    print("Assigning dot positions...")
    positioned = assign_dot_positions(daily, seed=RNG_SEED)
    positioned.to_csv(OUTPUT_DIR / "combined_animation_positioned_states.csv", index=False)

    print("Creating animation...")
    make_animation(positioned)

    print("Done.")
    print(f"Saved to: {OUTPUT_GIF}")


if __name__ == "__main__":
    main()