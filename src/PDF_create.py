import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#creates a funtion to import date data and parse it to ensure consitant formats 
def load_dates(path, date_cols):
    df = pd.read_csv(path)
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", format="%d/%m/%Y")
    return df
# creates funtion to move date to next specific weekday - for MDT dates 
def days_to_next_weekday(d: pd.Series, target_weekday: int, include_today=True) -> pd.Series:
    wd = d.dt.weekday
    if include_today:
        return (target_weekday - wd) % 7
    else:
        return ((target_weekday - (wd + 1) % 7) % 7) + 1

BIOPMDT_WD = 2   # Wednesday
TREATMDT_WD = 4  # Friday

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def build_pdfs(data_dir: Path = DATA_DIR):
#Import data 
    pre_referral = load_dates(
        #"/Users/louisefinlayson/Documents/Msc_proj/data/pre_refferal.csv",
        DATA_DIR / "pre_refferal.csv",
        ["Date of referral to pathway"]
    )

    pre_ref_to_mri = load_dates(
        DATA_DIR / "pre_ref_to_mri.csv",
        ["Date of referral to pathway", "Date of MRI"]
    )

    pre_mri_to_mrireport = load_dates(
        DATA_DIR / "pre_mri_to_mrirep.csv",
        ["Date of MRI", "Date MRI reported"]
    )

    pre_mrireport_to_biopmdt = load_dates(
        DATA_DIR / "pre_mrirep_to_biopmdt.csv",
        ["Date MRI reported", "Date of Prostate MRI MDT"]
    )

    pre_biopmdt_to_biop = load_dates(
        DATA_DIR / "pre_biopmdt_to_biop.csv",
        ["Date of Prostate MRI MDT", "Date of Biopsy"]
    )

    pre_biop_to_pathrep = load_dates(
        DATA_DIR / "pre_biop_to_pathrep.csv",
        ["Date of Biopsy", "Date of pathology report"]
    )

    pre_pathrep_to_treatmdt = load_dates(
        DATA_DIR / "pre_pathrep_to_treatmdt.csv",
        ["Date of pathology report", "Date of MDT (treatment options)"]
    )

    pre_treatmdt_to_outpat = load_dates(
        DATA_DIR / "pre_treatmdt_to_outpat.csv",
        ["Date of MDT (treatment options)", "Date of outpat appt"]
    )

    # Calculate durations 

    pre_ref_to_mri ["days_referral_to_mri"] = (
        pre_ref_to_mri ["Date of MRI"] - pre_ref_to_mri["Date of referral to pathway"]
    ).dt.days
    pdf_pre_ref_to_mri = pre_ref_to_mri["days_referral_to_mri"].dropna()

    pre_mri_to_mrireport ["days_mri_to_mrireport"] = (
        pre_mri_to_mrireport ["Date MRI reported"] - pre_mri_to_mrireport["Date of MRI"]
    ).dt.days
    pdf_pre_mri_to_mrireport = pre_mri_to_mrireport["days_mri_to_mrireport"].dropna()

    pre_mrireport_to_biopmdt ["days_mrireport_to_biomdt"] = (
        pre_mrireport_to_biopmdt["Date of Prostate MRI MDT"] - pre_mrireport_to_biopmdt["Date MRI reported"]
    ).dt.days
    pdf_pre_mrireport_to_biopmdt = pre_mrireport_to_biopmdt["days_mrireport_to_biomdt"].dropna()

    pre_biopmdt_to_biop ["days_biomdt_to_bio"] = (
        pre_biopmdt_to_biop ["Date of Biopsy"] - pre_biopmdt_to_biop ["Date of Prostate MRI MDT"]
    ).dt.days
    pdf_pre_biopmdt_to_biop = pre_biopmdt_to_biop ["days_biomdt_to_bio"].dropna()

    pre_biop_to_pathrep["days_bio_to_pathrep"] = (
        pre_biop_to_pathrep["Date of pathology report"] - pre_biop_to_pathrep["Date of Biopsy"]
    ).dt.days
    pdf_pre_biop_to_pathrep = pre_biop_to_pathrep["days_bio_to_pathrep"].dropna()

    pre_pathrep_to_treatmdt["days_pathrep_to_treatmdt"] = (
        pre_pathrep_to_treatmdt["Date of MDT (treatment options)"] - pre_pathrep_to_treatmdt["Date of pathology report"]
    ).dt.days
    pdf_pre_pathrep_to_treatmdt = pre_pathrep_to_treatmdt["days_pathrep_to_treatmdt"].dropna()

    pre_treatmdt_to_outpat ["days_treatmdt_to_outpat"] = (
        pre_treatmdt_to_outpat ["Date of outpat appt"] - pre_treatmdt_to_outpat ["Date of MDT (treatment options)"]
    ).dt.days
    pdf_pre_treatmdt_to_outpat = pre_treatmdt_to_outpat ["days_treatmdt_to_outpat"].dropna()

    #new PDFs for MDT dates
    # Example: MRI report -> Biopsy MDT
    df = pre_mrireport_to_biopmdt.dropna(subset=["Date MRI reported", "Date of Prostate MRI MDT"]).copy()

    # observed total wait
    df["w_obs"] = (df["Date of Prostate MRI MDT"] - df["Date MRI reported"]).dt.days
    # calendar component: days from report date to the next Wednesday
    df["s_obs"] = days_to_next_weekday(df["Date MRI reported"], BIOPMDT_WD, include_today=True)
    # queue component
    df["q_obs"] = df["w_obs"] - df["s_obs"]
    # keep only valid (non-negative) queue delays
    queue_pdf_mrirep_to_biopmdt = df.loc[df["q_obs"] >= 0, "q_obs"].astype(int)

    df2 = pre_pathrep_to_treatmdt.dropna(subset=["Date of pathology report", "Date of MDT (treatment options)"]).copy()
    df2["w_obs"] = (df2["Date of MDT (treatment options)"] - df2["Date of pathology report"]).dt.days
    df2["s_obs"] = days_to_next_weekday(df2["Date of pathology report"], TREATMDT_WD, include_today=True)
    df2["q_obs"] = df2["w_obs"] - df2["s_obs"]
    queue_pdf_pathrep_to_treatmdt = df2.loc[df2["q_obs"] >= 0, "q_obs"].astype(int)

    biopsy_residual_samples = pd.read_csv("biopsy_residual_samples_orig.csv") #file made in biop_queue_cal.py when no residual MC compnenet added to DES engine biop wait, biop delay set to 1 and capacity to {{3: 2, 4: 1}}
   #biopsy_residual_samples = pd.read_csv("biopsy_residual_samples.csv") #file made in biop_queue_cal.py when no residual MC compnenet added to DES engine biop wait, biop delay set to 1 and capacity to {{3: 2, 4: 1}}
    return {
        "pre_referral_to_mri": pdf_pre_ref_to_mri,
        "pre_mri_to_mrireport": pdf_pre_mri_to_mrireport,
        "pre_mrirep_to_biopsymdt": pdf_pre_mrireport_to_biopmdt,
        "pre_biopmdt_to_biop": pdf_pre_biopmdt_to_biop,
        "pre_biop_to_pathrep": pdf_pre_biop_to_pathrep,
        "pre_pathrep_to_treatmdt": pdf_pre_pathrep_to_treatmdt,
        "pre_treatmdt_to_outpat": pdf_pre_treatmdt_to_outpat,
        "queue_mrirep_to_biopsymdt": queue_pdf_mrirep_to_biopmdt,
        "queue_pathrep_to_treatmdt": queue_pdf_pathrep_to_treatmdt,
        "biopsy_residual_samples": biopsy_residual_samples
        
    }


#print(pdf_pre_treatmdt_to_outpat)

#plt.hist(pdf_pre_pathrep_to_treatmdt , bins=20)
#plt.xlabel("Days")
#plt.ylabel("Frequency")
#plt.title("Pathology Report → Treatment MDT waiting time")
#plt.show()

#print(pdf_pre_pathrep_to_treatmdt .describe())

#form ECDF using EQ.5 in project plan 

#x = np.sort(pdf_pre_ref_to_mri .values)
#ecdf = np.arange(1, len(x) + 1) / len(x)

#plt.step(x, ecdf, where="post")
#plt.xlabel("Days")
#plt.ylabel("ECDF")
#plt.title("ECDF: Referral → MRI")
#plt.show()


def build_branching(data_dir: Path = DATA_DIR):
    #calclualte decision node probablities 
    # Biopsy MDT decision
    # 0 = surveillance
    # 1 = biopsy
    # 2 = discharge
    biop_mdt_dec = pd.read_csv( DATA_DIR / "pre_biop_dec.csv")


    biop_mdt_dec["Outcome code"] = pd.to_numeric(biop_mdt_dec["Outcome code"], errors="coerce").astype("Int64")




    #biop_dec_branch_probs = biop_mdt_dec ["Outcome code"].value_counts(normalize=True).to_dict()

    biop_dec_branch_probs = biop_mdt_dec["Outcome code"].value_counts(normalize=True).dropna().to_dict()
    biop_dec_branch_probs = {int(k): float(v) for k, v in biop_dec_branch_probs.items()}

    #print(biop_dec_branch_probs)

    # biopsy outcome
    # 0 = no pathology found
    # 1 = Cancer

    path_dec = pd.read_csv( DATA_DIR / "pre_pathrep_outcome.csv")

   
    path_dec["Outcome code"] = pd.to_numeric(path_dec["Outcome code"], errors="coerce").astype("Int64")

    #path_dec_branch_probs = path_dec ["Outcome code"].value_counts(normalize=True).to_dict()
    path_dec_branch_probs = path_dec["Outcome code"].value_counts(normalize=True).dropna().to_dict()
    path_dec_branch_probs = {int(k): float(v) for k, v in path_dec_branch_probs.items()}



    

    return {
        "biopmdt_outcome": biop_dec_branch_probs,
        "pathrep_outcome": path_dec_branch_probs,
    }
#print(path_dec_branch_probs)

if __name__ == "__main__":
    pdfs = build_pdfs()
    branching = build_branching()
    print("Built PDFs:", list(pdfs.keys()))
    print("Built branching sets:", list(branching.keys()))

# save PDFs and branching proabilities into project directory
# “To ensure modularity and transparency, empirical waiting-time 
# distributions and branching probabilities were derived 
# in standalone preprocessing scripts and imported into the simulation as independent inputs.”
#pdfs = {
 #   "pre_referral_to_mri": pdf_pre_ref_to_mri,
  #  "pre_mri_to_mrireport": pdf_pre_mri_to_mrireport,
   # "pre_mrirep_to_biopsymdt": pdf_pre_mrireport_to_biopmdt,
    #"pre_biopmdt_to_biop": pdf_pre_biopmdt_to_biop,
  #  "pre_biop_to_pathrep": pdf_pre_biop_to_pathrep,
   # "pre_pathrep_to_treatmdt": pdf_pre_pathrep_to_treatmdt,
    #"pre_treatmdt_to_outpat": pdf_pre_treatmdt_to_outpat
#}

#branching = {
 #   "biopmdt_outcome": biop_dec_branch_probs,
  #   "pathrep_outcome": path_dec_branch_probs
#}



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def build_pdfs2(exclude_np053_ref_to_mri: bool = True):
    # Load files
    pre_ref_to_mri = pd.read_csv(DATA_DIR / "pre_ref_to_mri.csv")
    pre_mri_to_mrireport = pd.read_csv(DATA_DIR / "pre_mri_to_mrirep.csv")
    pre_mrirep_to_biopsymdt = pd.read_csv(DATA_DIR / "pre_mrirep_to_biopmdt.csv")
    pre_biopmdt_to_biop = pd.read_csv(DATA_DIR / "pre_biopmdt_to_biop.csv")
    pre_biop_to_pathrep = pd.read_csv(DATA_DIR / "pre_biop_to_pathrep.csv")
    pre_pathrep_to_treatmdt = pd.read_csv(DATA_DIR / "pre_pathrep_to_treatmdt.csv")
    pre_treatmdt_to_outpat = pd.read_csv(DATA_DIR / "pre_treatmdt_to_outpat.csv")

    

    # Optional sensitivity exclusion
    if exclude_np053_ref_to_mri:
        pre_ref_to_mri = pre_ref_to_mri.loc[
            pre_ref_to_mri["Subject number"] != "NP053"
        ].copy()

    pdfs = {
        "pre_referral_to_mri": (
            pd.to_datetime(pre_ref_to_mri["Date of MRI"], dayfirst=True, errors="coerce")
            - pd.to_datetime(pre_ref_to_mri["Date of referral to pathway"], dayfirst=True, errors="coerce")
        ).dt.days.dropna(),

        "pre_mri_to_mrireport": (
            pd.to_datetime(pre_mri_to_mrireport["Date MRI reported"], dayfirst=True, errors="coerce")
            - pd.to_datetime(pre_mri_to_mrireport["Date of MRI"], dayfirst=True, errors="coerce")
        ).dt.days.dropna(),

        "pre_mrirep_to_biopsymdt": (
            pd.to_datetime(pre_mrirep_to_biopsymdt["Date of Prostate MRI MDT"], dayfirst=True, errors="coerce")
            - pd.to_datetime(pre_mrirep_to_biopsymdt["Date MRI reported"], dayfirst=True, errors="coerce")
        ).dt.days.dropna(),

        "pre_biopmdt_to_biop": (
            pd.to_datetime(pre_biopmdt_to_biop["Date of Biopsy"], dayfirst=True, errors="coerce")
            - pd.to_datetime(pre_biopmdt_to_biop["Date of Prostate MRI MDT"], dayfirst=True, errors="coerce")
        ).dt.days.dropna(),

        "pre_biop_to_pathrep": (
            pd.to_datetime(pre_biop_to_pathrep["Date of pathology report"], dayfirst=True, errors="coerce")
            - pd.to_datetime(pre_biop_to_pathrep["Date of Biopsy"], dayfirst=True, errors="coerce")
        ).dt.days.dropna(),

        "pre_pathrep_to_treatmdt": (
            pd.to_datetime(pre_pathrep_to_treatmdt["Date of MDT (treatment options)"], dayfirst=True, errors="coerce")
            - pd.to_datetime(pre_pathrep_to_treatmdt["Date of pathology report"], dayfirst=True, errors="coerce")
        ).dt.days.dropna(),

        "pre_treatmdt_to_outpat": (
            pd.to_datetime(pre_treatmdt_to_outpat["Date of outpat appt"], dayfirst=True, errors="coerce")
            - pd.to_datetime(pre_treatmdt_to_outpat["Date of MDT (treatment options)"], dayfirst=True, errors="coerce")
        ).dt.days.dropna(),


    }

    return pdfs
