from pathlib import Path
from data_prep.empirical_inputs import build_branching

DATA_DIR = Path("data")

branching = build_branching(DATA_DIR)

print("\n=== Biopsy MDT outcome probabilities ===")
for outcome, prob in branching["biopmdt_outcome"].items():
    print(f"Outcome {outcome}: {prob:.3f} ({prob*100:.1f}%)")

print("\n=== Pathology outcome probabilities ===")
for outcome, prob in branching["pathrep_outcome"].items():
    print(f"Outcome {outcome}: {prob:.3f} ({prob*100:.1f}%)")