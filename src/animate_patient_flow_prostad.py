from pathlib import Path
from datetime import timedelta
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

EVENT_LOG_PATH = OUTPUT_DIR / "prostad_events.csv"
#EVENT_LOG_PATH = OUTPUT_DIR / "prostad_events.csv"

# Change this if you want fewer/more dots on screen
MAX_PATIENTS_TO_ANIMATE = 250

# Animation output
OUTPUT_GIF = OUTPUT_DIR / "patient_flow_animation_prostad.gif"
OUTPUT_MP4 = OUTPUT_DIR / "patient_flow_animation_prostad.mp4"

# Frame timing
FPS = 4  # 4 frames per second = 1 day per frame

# Random seed for stable dot positions between runs
RNG_SEED = 1234


# ============================================================
# EVENT DEFINITIONS
# ============================================================

# These are the milestone events expected in your batch_events.csv
# They should match the names used in your simulation output.
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

# Human-readable labels for plotting
STATE_LABELS = {
    "queue_ref_to_mri": "Referral → MRI",
    "in_mri": "MRI",
    "queue_mri_to_report": "MRI → Report",
    "in_reporting": "MRI Report",
    "queue_report_to_biopmdt": "Report → Biopsy MDT",
    "in_biopmdt": "Biopsy MDT",
    "queue_biopmdt_to_biopsy": "MDT → Biopsy",
    "in_biopsy": "Biopsy",
    "queue_biopsy_to_pathrep": "Biopsy → Path Report",
    "in_pathrep": "Path Report",
    "queue_pathrep_to_treatmdt": "Path Report → Treat MDT",
    "in_treatmdt": "Treatment MDT",
    "queue_treatmdt_to_outpat": "Treat MDT → Outpatient",
    "in_outpatient": "Outpatient",
    "exit_after_biopmdt": "Discharged / Surveillance",
    "exit_after_pathrep": "No Pathology Found",
    "exit_after_outpatient": "Pathway Completed",
    #"completed": "Completed / Exit",
}

# Layout positions for each state box
# x, y, width, height
STATE_LAYOUT = {
    "queue_ref_to_mri": (0.5, 4.5, 2.2, 1.8),
    #"in_mri": (3.2, 5.0, 1.4, 1.4),
    #"queue_mri_to_report": (5.1, 5.0, 2.2, 1.4),
    "queue_mri_to_report": (4.42, 4.5, 2.2, 1.8),
    #"in_reporting": (7.8, 5.0, 1.4, 1.4),

    #"queue_report_to_biopmdt": (10.0, 5.0, 2.4, 1.4),
    "queue_report_to_biopmdt": (8.34, 4.5, 2.2, 1.8),
    #"in_biopmdt": (12.9, 5.0, 1.6, 1.4),

    #"queue_biopmdt_to_biopsy": (10.0, 2.7, 2.4, 1.4),
    "queue_biopmdt_to_biopsy": (12.26, 4.5, 2.2, 1.8),
    #"in_biopsy": (12.9, 2.7, 1.6, 1.4),

    #"queue_biopsy_to_pathrep": (15.2, 2.7, 2.4, 1.4),
    "queue_biopsy_to_pathrep": (16.18, 4.5, 2.2, 1.8),
    #"in_pathrep": (18.1, 2.7, 1.6, 1.4),

    #"queue_pathrep_to_treatmdt": (15.2, 5.0, 2.4, 1.4),
    "queue_pathrep_to_treatmdt": (20.1, 4.5, 2.2, 1.8),
    #"in_treatmdt": (18.1, 5.0, 1.6, 1.4),

    #"queue_treatmdt_to_outpat": (20.5, 5.0, 2.4, 1.4),
    "queue_treatmdt_to_outpat": (24.0, 4.5, 2.2, 1.8),
    #"in_outpatient": (23.4, 5.0, 1.6, 1.4),

    #"completed": (26.0, 4.0, 2.0, 2.0),
    "exit_after_biopmdt": (8.34, 2.01, 2.2, 1.8),
    "exit_after_pathrep": (16.18, 2.01, 2.2, 1.8),
    "exit_after_outpatient": (24.0, 2.01, 2.2, 1.8),
}

OUTCOME_COLOUR_MAP = {
    "exit_after_outpatient": "tab:blue",
    "exit_after_pathrep": "tab:orange",
    "exit_after_biopmdt": "tab:red",
    "exit_unknown": "tab:gray",
    "unknown": "tab:gray",
}

EXIT_BOX_FACECOLOURS = {
    "exit_after_biopmdt": "mistyrose",
    "exit_after_pathrep": "moccasin",
    "exit_after_outpatient": "lightblue",
}


# ============================================================
# HELPERS
# ============================================================

def load_event_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"patient_id", "event", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Event log missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Keep only events we know how to use
    df = df[df["event"].isin(EVENT_ORDER)].copy()

    # If duplicate event rows exist for a patient/event, keep earliest
    df = (
        df.sort_values(["patient_id", "event", "date"])
          .drop_duplicates(subset=["patient_id", "event"], keep="first")
          .reset_index(drop=True)
    )

    return df


def pivot_patient_events(df: pd.DataFrame) -> pd.DataFrame:
    wide = (
        df.pivot(index="patient_id", columns="event", values="date")
          .reset_index()
    )

    # Ensure all expected columns exist
    for ev in EVENT_ORDER:
        if ev not in wide.columns:
            wide[ev] = pd.NaT

    return wide


def sample_patients(wide: pd.DataFrame, max_patients: int, seed: int) -> pd.DataFrame:
    if len(wide) <= max_patients:
        return wide.copy()

    rng = np.random.default_rng(seed)
    keep_ids = rng.choice(wide["patient_id"].to_numpy(), size=max_patients, replace=False)
    return wide[wide["patient_id"].isin(keep_ids)].copy()


def daterange(start_date: pd.Timestamp, end_date: pd.Timestamp):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def add_interval(records, patient_id, start_date, end_date, state):
    """
    Add one record per day for [start_date, end_date], inclusive.
    """
    if pd.isna(start_date) or pd.isna(end_date):
        return
    if end_date < start_date:
        return

    for d in daterange(start_date, end_date):
        records.append(
            {
                "patient_id": patient_id,
                "date": d,
                "state": state,
            }
        )

def assign_patient_outcome_colours(daily_states: pd.DataFrame) -> dict:
    """
    Assign one stable colour per patient based on their eventual exit state.
    """
    patient_colour = {}

    for patient_id, grp in daily_states.groupby("patient_id"):
        states = grp.sort_values("date")["state"].tolist()

        final_exit = None
        for s in reversed(states):
            if s in {"exit_after_outpatient", "exit_after_pathrep", "exit_after_biopmdt", "exit_unknown"}:
                final_exit = s
                break

        if final_exit is None:
            final_exit = "unknown"

        patient_colour[patient_id] = OUTCOME_COLOUR_MAP.get(final_exit, "tab:gray")

    return patient_colour


def build_daily_states(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert patient milestone dates into one row per patient per day.
    Assumptions:
    - Queue states occupy days between milestone events.
    - Instantaneous milestone events occupy their event date.
    - If a downstream event is missing, patient moves to 'completed'
      after their last observed milestone.
    """
    records = []

    all_dates = []
    for ev in EVENT_ORDER:
        if ev in wide.columns:
            all_dates.extend(wide[ev].dropna().tolist())

    if not all_dates:
        raise ValueError("No valid event dates found in event log.")

    global_start = min(all_dates)
    global_end = max(all_dates) + timedelta(days=5)

    for _, row in wide.iterrows():
        pid = row["patient_id"]

        ref = row["referral_received"]
        mri = row["mri_performed"]
        rep = row["mri_report_ready"]
        bmdt = row["MDT_occured"]
        biopsy = row["biopsy_done"]
        pathrep = row["Path_report_recieved"]
        treat = row["Treatment_options_MDT_occured"]
        outpat = row["Outpatient_appointment_occured"]

        # -------------------------
        # Referral -> MRI
        # -------------------------
        if pd.notna(ref):
            if pd.notna(mri):
                add_interval(records, pid, ref, mri - timedelta(days=1), "queue_ref_to_mri")
                #add_interval(records, pid, mri, mri, "in_mri")
            else:
                add_interval(records, pid, ref, global_end, "queue_ref_to_mri")
                continue

        # -------------------------
        # MRI -> Report
        # -------------------------
        if pd.notna(mri):
            if pd.notna(rep):
                add_interval(records, pid, mri + timedelta(days=1), rep - timedelta(days=1), "queue_mri_to_report")
               # add_interval(records, pid, rep, rep, "in_reporting")
            else:
                add_interval(records, pid, mri + timedelta(days=1), global_end, "queue_mri_to_report")
                continue

        # -------------------------
        # Report -> Biopsy MDT
        # -------------------------
        if pd.notna(rep):
            if pd.notna(bmdt):
                add_interval(records, pid, rep + timedelta(days=1), bmdt - timedelta(days=1), "queue_report_to_biopmdt")
                #add_interval(records, pid, bmdt, bmdt, "in_biopmdt")
            else:
                # fallback if no biopsy MDT event exists
                add_interval(records, pid, rep + timedelta(days=1), global_end, "exit_after_biopmdt")
                continue

        # -------------------------
        # Branch 1: biopsy pathway
        # -------------------------
        if pd.notna(bmdt) and pd.notna(biopsy):
            add_interval(records, pid, bmdt + timedelta(days=1), biopsy - timedelta(days=1), "queue_biopmdt_to_biopsy")
            #add_interval(records, pid, biopsy, biopsy, "in_biopsy")

            if pd.notna(pathrep):
                add_interval(records, pid, biopsy + timedelta(days=1), pathrep - timedelta(days=1), "queue_biopsy_to_pathrep")
                #add_interval(records, pid, pathrep, pathrep, "in_pathrep")
            else:
                add_interval(records, pid, biopsy + timedelta(days=1), global_end, "queue_biopsy_to_pathrep")
                continue

            if pd.notna(treat):
                add_interval(records, pid, pathrep + timedelta(days=1), treat - timedelta(days=1), "queue_pathrep_to_treatmdt")
                #add_interval(records, pid, treat, treat, "in_treatmdt")

                if pd.notna(outpat):
                    add_interval(records, pid, treat + timedelta(days=1), outpat - timedelta(days=1), "queue_treatmdt_to_outpat")
                   # add_interval(records, pid, outpat, outpat, "in_outpatient")
                    add_interval(records, pid, outpat + timedelta(days=1), global_end, "exit_after_outpatient")
                else:
                    add_interval(records, pid, treat + timedelta(days=1), global_end, "queue_treatmdt_to_outpat")
            else:
                # no treatment MDT after pathology = exits after pathology
                add_interval(records, pid, pathrep + timedelta(days=1), global_end, "exit_after_pathrep")

        # -------------------------
        # Branch 2: no biopsy after biopsy MDT
        # -------------------------
        elif pd.notna(bmdt) and pd.isna(biopsy):
            add_interval(records, pid, bmdt + timedelta(days=1), global_end, "exit_after_biopmdt")

    daily = pd.DataFrame(records).drop_duplicates(subset=["patient_id", "date"], keep="last")
    daily = daily.sort_values(["date", "patient_id"]).reset_index(drop=True)
    return daily


def assign_dot_positions(daily: pd.DataFrame, seed: int = 1234) -> pd.DataFrame:
    """
    Assign stable positions inside each state box for each patient-day.
    """
    rng = np.random.default_rng(seed)
    positioned_frames = []

    for date_value, day_df in daily.groupby("date"):
        day_df = day_df.copy()

        xs = []
        ys = []

        for state, state_df in day_df.groupby("state"):
            if state not in STATE_LAYOUT:
                raise ValueError(f"State '{state}' missing from STATE_LAYOUT")

            x, y, w, h = STATE_LAYOUT[state]
            n = len(state_df)

            # Grid-ish packing inside the box
            cols = max(1, math.ceil(math.sqrt(n)))
            rows = max(1, math.ceil(n / cols))

            x_margin = 0.12
            y_margin = 0.12
            usable_w = max(0.1, w - 2 * x_margin)
            usable_h = max(0.1, h - 2 * y_margin)

            x_step = usable_w / max(cols, 1)
            y_step = usable_h / max(rows, 1)

            coords = []
            for i in range(n):
                col = i % cols
                row = i // cols

                px = x + x_margin + (col + 0.5) * x_step
                py = y + y_margin + usable_h - (row + 0.5) * y_step

                # Tiny jitter so dots do not look too rigid
                px += rng.uniform(-0.03, 0.03)
                py += rng.uniform(-0.03, 0.03)
                coords.append((px, py))

            state_df = state_df.sort_values("patient_id").copy()
            state_df["x"] = [c[0] for c in coords]
            state_df["y"] = [c[1] for c in coords]
            positioned_frames.append(state_df)

    out = pd.concat(positioned_frames, ignore_index=True)
    return out


def draw_static_layout(ax):
    for state, (x, y, w, h) in STATE_LAYOUT.items():
        facecolor = EXIT_BOX_FACECOLOURS.get(state, "none")
        rect = Rectangle((x, y), w, h, fill=True, facecolor=facecolor, edgecolor = "black", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h + 0.12,
            STATE_LABELS.get(state, state),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Simple connecting arrows as text markers
    ax.annotate("", xy=(4.0, 5.4), xytext=(3.0, 5.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.92, 5.4), xytext=(6.92, 5.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(11.84, 5.4), xytext=(10.84, 5.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(15.76, 5.4), xytext=(14.76, 5.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(19.68, 5.4), xytext=(18.68, 5.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(23.6, 5.4), xytext=(22.6, 5.4), arrowprops=dict(arrowstyle="->"))




    

    ax.set_xlim(0, 29)
    ax.set_ylim(2, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")


def make_animation(positioned_daily: pd.DataFrame, save_gif=True, save_mp4=False):
    dates = sorted(positioned_daily["date"].dropna().unique())
    if len(dates) == 0:
        raise ValueError("No daily positions available for animation.")

    fig, ax = plt.subplots(figsize=(18, 6))
    scatter = ax.scatter([], [], s=20, alpha=0.75)
    day_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")
    count_text = ax.text(0.02, 0.90, "", transform=ax.transAxes, ha="left", va="top", fontsize=10)

    def init():
        ax.clear()
        draw_static_layout(ax)
        return scatter,

    patient_colour_map = assign_patient_outcome_colours(positioned_daily)

    def update(frame_idx):
        ax.clear()
        draw_static_layout(ax)

        current_date = dates[frame_idx]
        frame_df = positioned_daily[positioned_daily["date"] == current_date].copy()

        frame_colours = frame_df["patient_id"].map(patient_colour_map).fillna("tab:gray")

        ax.scatter(
            frame_df["x"],
            frame_df["y"],
            s=20,
            alpha=0.8,
            c=frame_colours
        )

        day_number = frame_idx
        ax.text(
            0.0, 0.96,
            f"Day {day_number}   |   Date: {pd.Timestamp(current_date).date()}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=14, fontweight="bold"
        )

        state_counts = frame_df["state"].value_counts().sort_index()
        summary = "   |   ".join([f"{STATE_LABELS.get(s, s).replace(chr(10), ' ')}: {n}" for s, n in state_counts.items()])
        #ax.text(
         #   0.02, 0.90,
          #  summary,
           # transform=ax.transAxes,
            #ha="left", va="top",
           # fontsize=9
        #)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Exit after outpatient',
                    markerfacecolor='tab:blue', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Exit after pathology',
                    markerfacecolor='tab:orange', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Exit after biopsy MDT',
                    markerfacecolor='tab:red', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Unknown / unresolved',
                    markerfacecolor='tab:gray', markersize=8),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.6, 1.0),
            ncol=4,
            frameon=True,
            fontsize=9
        )

        return scatter,

    anim = FuncAnimation(
        fig,
        update,
        frames=len(dates),
        init_func=init,
        interval=1000 / FPS,
        blit=False,
        repeat=False,
    )

    if save_gif:
        print(f"Saving GIF to {OUTPUT_GIF} ...")
        anim.save(OUTPUT_GIF, writer=PillowWriter(fps=FPS))

    if save_mp4:
        print(f"Saving MP4 to {OUTPUT_MP4} ...")
        anim.save(OUTPUT_MP4, writer="ffmpeg", fps=FPS)

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading event log...")
    event_log = load_event_log(EVENT_LOG_PATH)

    print("Pivoting patient milestones...")
    patient_wide = pivot_patient_events(event_log)

    print(f"Patients in log: {len(patient_wide)}")
    patient_wide = sample_patients(
        patient_wide,
        max_patients=MAX_PATIENTS_TO_ANIMATE,
        seed=RNG_SEED,
    )
    print(f"Patients selected for animation: {len(patient_wide)}")

    print("Building daily states...")
    daily = build_daily_states(patient_wide)

    # Save the inferred daily states too, useful for debugging
    daily_out = OUTPUT_DIR / "patient_daily_states.csv"
    daily.to_csv(daily_out, index=False)
    print(f"Daily states saved to: {daily_out}")

    print("Assigning dot positions...")
    positioned = assign_dot_positions(daily, seed=RNG_SEED)

    print("Making animation...")
    make_animation(
        positioned_daily=positioned,
        save_gif=True,
        save_mp4=False,   # change to True if you have ffmpeg installed
    )

    print("Done.")


if __name__ == "__main__":
    main()