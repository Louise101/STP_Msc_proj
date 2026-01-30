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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

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

#calclualte decision node probablities 


# Biopsy MDT decision
# 0 = surveillance
# 1 = biopsy
# 2= discharge
biop_mdt_dec = pd.read_csv( DATA_DIR / "pre_biop_dec.csv")

biop_dec_branch_probs = biop_mdt_dec ["Outcome code"].value_counts(normalize=True)
#print(biop_dec_branch_probs)

# biopsy outcome
# 0 = no pathology found

path_dec = pd.read_csv( DATA_DIR / "pre_pathrep_outcome.csv")

path_dec_branch_probs = path_dec ["Outcome code"].value_counts(normalize=True)
#print(path_dec_branch_probs)

# save PDFs and branching proabilities into project directory
pdfs = {
    "pre_referral_to_mri": pdf_pre_ref_to_mri,
    "pre_mri_to_mrireport": pdf_pre_mri_to_mrireport,
    "pre_mrirep_to_biopsymdt": pdf_pre_mrireport_to_biopmdt,
    "pre_biopmdt_to_biop": pdf_pre_biopmdt_to_biop,
    "pre_biop_to_pathrep": pdf_pre_biop_to_pathrep,
    "pre_pathrep_to_treatmdt": pdf_pre_pathrep_to_treatmdt,
    "pre_treatmdt_to_outpat": pdf_pre_treatmdt_to_outpat
}

branching = {
    "biopmdt_outcome": biop_dec_branch_probs,
     "pathrep_outcome": path_dec_branch_probs
}
