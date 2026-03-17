import datetime as dt
import numpy as np
import sys
from typing import Optional, Dict, Any

from PDF_create import build_pdfs, build_branching
from sampling import sample_empirical_ecdf, sample_outcome,sample_mri_to_report_correlated



#function to restric pathway progress on weekends 
def next_weekday(d: dt.date) -> dt.date:
    # 0=Mon ... 6=Sun
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d

# fuction to move MDTs to specific weekdays
def next_allowed_weekday(d: dt.date, allowed_weekdays: set[int]) -> dt.date:
    """
    Move date forward until it lands on one of the allowed weekdays.
    allowed_weekdays: set of ints where 0=Mon ... 6=Sun
    """
    if not allowed_weekdays:
        raise ValueError("allowed_weekdays must be non-empty")
    while d.weekday() not in allowed_weekdays:
        d += dt.timedelta(days=1)
    return d

BIOPMDT_DAYS = {2}   # Wednesday
TREATMDT_DAYS = {4}  # Friday
#def trace_one_patient(start_date: dt.date, rng: np.random.Generator, patient_id: str = "VP0001"):
def trace_one_patient_mdtday(start_date: dt.date, rng, pdfs: Optional[dict] = None, branching: Optional[dict] = None, patient_id="VP0001"):

   # u_patient = rng.random()
   # ALPHA = 0.4  # try 0.4 first; tune later

    pdfs = build_pdfs()
    branching = build_branching()
   


    log = []
    #patient_id = "VP0001"

    # Step 1: Referral
    referral_date = start_date
    log.append({"patient_id": patient_id, "event": "referral_received", "date": referral_date})

    # Step 2: Referral -> MRI
    #t_ref_to_mri = sample_empirical_ecdf(pdfs["pre_referral_to_mri"], rng=rng)
    #from sampling import correlated_u
    #u = correlated_u(u_patient, rng=rng, alpha=ALPHA)
    t_ref_to_mri = sample_empirical_ecdf(pdfs["pre_referral_to_mri"], rng=rng)
    mri_date_raw = referral_date + dt.timedelta(days=int(t_ref_to_mri))
    mri_date = next_weekday(mri_date_raw)  # apply weekday constraint 
    log.append({"patient_id": patient_id, "event": "mri_performed", "date": mri_date, "wait_days": int(t_ref_to_mri)})

    # Step 3: MRI -> Report
   # t_mri_to_report = sample_empirical_ecdf(pdfs["pre_mri_to_mrireport"], rng=rng)
    #from sampling import correlated_u
    #u = correlated_u(u_patient, rng=rng, alpha=ALPHA)
   # t_mri_to_report = sample_empirical_ecdf(pdfs["pre_mri_to_mrireport"], rng=rng)
    t_mri_to_report = sample_mri_to_report_correlated(
    ref_to_mri_wait=t_ref_to_mri,
    ref_to_mri_samples=pdfs["pre_referral_to_mri"],
    mri_to_report_samples=pdfs["pre_mri_to_mrireport"],
    rng=rng,
    gaussian_corr=0.2)

    report_date_raw = mri_date + dt.timedelta(days=int(t_mri_to_report))
    report_date = next_weekday(report_date_raw)  # apply weekday constraint 
    log.append({"patient_id": patient_id, "event": "mri_report_ready", "date": report_date, "wait_days": int(t_mri_to_report)})

    # Step 4: Report -> MDT

    #q= sample_empirical_ecdf(pdfs["queue_mrirep_to_biopsymdt"], rng=rng)
    #ready_date = report_date + dt.timedelta(days=int(q))
    #MDT_date = next_allowed_weekday(ready_date, {2})  # Wed
    #wait_effective = (MDT_date - report_date).days
   # log.append({
      #  "patient_id": patient_id,
     #   "event": "MDT_occured",
    #    "date": MDT_date,
   #     "wait_days": wait_effective
   # })
    t_report_to_MDT = sample_empirical_ecdf(pdfs["pre_mrirep_to_biopsymdt"], rng=rng)
    MDT_date_raw = report_date +  dt.timedelta(days=int(t_report_to_MDT ))
    MDT_date = next_weekday(MDT_date_raw)  # apply weekday constraint  - !!!NEED TO FIND SPECIFIC MDT DAY!!!
    log.append({"patient_id": patient_id, "event": "MDT_occured", "date": MDT_date, "wait_days": int(t_report_to_MDT)})

    # Step 5: MDT outcome (branch)
    outcome = sample_outcome(branching["biopmdt_outcome"], rng=rng)
    #print("biop keys:", branching["biopmdt_outcome"].keys())
    outcome = int(outcome)
    log.append({"patient_id": patient_id, "event": "mdt_decision", "date": MDT_date, "outcome": outcome})

    # Step 6: Continue depending on outcome
    if outcome == 1: # biopsy # MDT -> Biopsy
        #t_mdt_to_biopsy = sample_empirical_ecdf(pdfs["pre_biopmdt_to_biop"], rng=rng)
       # u = correlated_u(u_patient, rng=rng, alpha=ALPHA)
        t_mdt_to_biopsy = sample_empirical_ecdf(pdfs["pre_biopmdt_to_biop"], rng=rng)
        biopsy_date_raw = MDT_date + dt.timedelta(days=int(t_mdt_to_biopsy))
        biopsy_date = next_weekday(biopsy_date_raw)
        log.append({"patient_id": patient_id, "event": "biopsy_done", "date": biopsy_date, "wait_days": int(t_mdt_to_biopsy)})
        #end_date = biopsy_date
    else:
        # discharge / surveillance etc
        end_date = MDT_date 
        log.append({"patient_id": patient_id, "event": "pathway_exit", "date": end_date})
        total_days = (end_date - referral_date).days
        return log, total_days

    # Step 7: Biopsy -> Path Report
   # t_biopsy_to_pathreport = sample_empirical_ecdf(pdfs["pre_biop_to_pathrep"], rng=rng)
    #u = correlated_u(u_patient, rng=rng, alpha=ALPHA)
    t_biopsy_to_pathreport = sample_empirical_ecdf(pdfs["pre_biop_to_pathrep"], rng=rng)
    pathrep_date_raw = biopsy_date + dt.timedelta(days=int(t_biopsy_to_pathreport))
    pathrep_date = next_weekday(pathrep_date_raw)  # apply weekday constraint 
    log.append({"patient_id": patient_id, "event": "Path_report_recieved", "date": pathrep_date, "wait_days": int(t_biopsy_to_pathreport)})


    # Step 8: Path report outcome (branch)
    path_outcome = sample_outcome(branching["pathrep_outcome"], rng=rng)
    #print("path keys:", branching["pathrep_outcome"].keys())
    path_outcome = int(path_outcome)
    log.append({"patient_id": patient_id, "event": "Path_report_outcome", "date": pathrep_date, "outcome": path_outcome})

    # step 9: Continue depending on outcome : Path report -> Treatment MDT 
    if path_outcome == 1: # cancer # Path report -> Treatment MDT 
       # u = correlated_u(u_patient, rng=rng, alpha=ALPHA)
        #q = sample_empirical_ecdf(pdfs["queue_pathrep_to_treatmdt"], rng=rng)
        #ready_date = pathrep_date + dt.timedelta(days=int(q))
        #treatMDT_date = next_allowed_weekday(ready_date, {4})  # Fri
        #wait_effective = (treatMDT_date - pathrep_date).days
        #log.append({
        #    "patient_id": patient_id,
          #  "event": "Treatment_options_MDT_occured",
         #   "date": treatMDT_date,
          #  "wait_days": wait_effective
        #})
        t_pathrep_to_treatMDT = sample_empirical_ecdf(pdfs["pre_pathrep_to_treatmdt"], rng=rng)
        treatMDT_date_raw = pathrep_date + dt.timedelta(days=int(t_pathrep_to_treatMDT))
        treatMDT_date = next_weekday(treatMDT_date_raw) # !!! GET TEATMENT MDT DAYS!!!
        #treatMDT_date_raw = pathrep_date + dt.timedelta(days=int(t_pathrep_to_treatMDT))
       # treatMDT_date = next_allowed_weekday(treatMDT_date_raw, TREATMDT_DAYS)
        log.append({"patient_id": patient_id, "event": "Treatment_options_MDT_occured", "date": treatMDT_date, "wait_days": int(t_pathrep_to_treatMDT)})
        #log.append({
        #"patient_id": patient_id,
        #"event": "MDT_occured",
       # "date": treatMDT_date,
      #  "wait_days_sampled": int(t_pathrep_to_treatMDT),
     #   "wait_days": (treatMDT_date - pathrep_date).days,
    #})

        #treatMDT_date = next_mdt_day(pathrep_date, TREATMDT_WEEKDAY, include_today=True)
        #wait = (treatMDT_date - pathrep_date).days
        #log.append({"patient_id": patient_id, "event": "Treatment_options_MDT_occured", "date": treatMDT_date, "wait_days": wait})
    else:
        # 0 - no pathology found
        end_date = pathrep_date
        log.append({"patient_id": patient_id, "event": "pathway_exit", "date": end_date})
        total_days = (end_date - referral_date).days
        return log, total_days

    # Step 10: TreatmentMDT -> Outpatient appointment 
   # u = correlated_u(u_patient, rng=rng, alpha=ALPHA)
    t_treatMDT_to_outpat = sample_empirical_ecdf(pdfs["pre_treatmdt_to_outpat"], rng=rng)
    outpat_date_raw = treatMDT_date + dt.timedelta(days=int(t_treatMDT_to_outpat))
    outpat_date = next_weekday(outpat_date_raw)  # apply weekday constraint 
    log.append({"patient_id": patient_id, "event": "Outpatient_appointment_occured", "date": outpat_date, "wait_days": int(t_treatMDT_to_outpat)})

    
    #step 11: end pathway 
    end_date = outpat_date
    log.append({"patient_id": patient_id, "event": "pathway_end", "date": end_date})
    

    total_days = (end_date - referral_date).days
    return log, total_days


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    log, total_days = trace_one_patient_mdtday(dt.date(2026, 1, 5), rng)

    for e in log:
        #print(e)
        print(e["event"], e.get("date"))
    print("Total pathway days:", total_days)
