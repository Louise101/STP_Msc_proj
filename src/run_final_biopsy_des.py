from config_final_biopsy_des import FINAL_BIOPSY_DES_CFG
from des_engine import run_day_loop_with_stage_engine
from single_walk_mdt_day import trace_one_patient_mdtday


def main():
    results = run_day_loop_with_stage_engine(FINAL_BIOPSY_DES_CFG, trace_one_patient_mdtday)
    print(results["summary_stats"])


if __name__ == "__main__":
    main()