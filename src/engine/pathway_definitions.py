from __future__ import annotations

from typing import Any


WAIT_MODE_MC = "MC"
WAIT_MODE_DES = "DES"


STAGE_CONFIG: dict[str, dict[str, Any]] = {
    "ref_to_mri": {
        "resource": "MRI_PROSTAD",
        "pdf_key": "pre_referral_to_mri",
        "completion_event": "mri_performed",
    },
    "mri_to_report": {
        "resource": None,
        "pdf_key": "pre_mri_to_mrireport",
        "completion_event": "mri_report_ready",
    },
    "report_to_biopmdt": {
        "resource": None,
        "pdf_key": "pre_mrirep_to_biopsymdt",
        "completion_event": "MDT_occured",
    },
    "biopmdt_to_biopsy": {
        "resource": None,
        "pdf_key": "pre_biopmdt_to_biop",
        "completion_event": "biopsy_done",
    },
    "biopsy_to_pathrep": {
        "resource": None,
        "pdf_key": "pre_biop_to_pathrep",
        "completion_event": "Path_report_recieved",
    },
    "pathrep_to_treatmdt": {
        "resource": None,
        "pdf_key": "pre_pathrep_to_treatmdt",
        "completion_event": "Treatment_options_MDT_occured",
    },
    "treatmdt_to_outpat": {
        "resource": None,
        "pdf_key": "pre_treatmdt_to_outpat",
        "completion_event": "Outpatient_appointment_occured",
    },
}

WAIT_STREAM_BY_STAGE = {
    "ref_to_mri": "wait_ref_to_mri",
    "mri_to_report": "wait_mri_to_report",
    "report_to_biopmdt": "wait_report_to_biopmdt",
    "biopmdt_to_biopsy": "wait_biopmdt_to_biopsy",
    "biopsy_to_pathrep": "wait_biopsy_to_pathrep",
    "pathrep_to_treatmdt": "wait_pathrep_to_treatmdt",
    "treatmdt_to_outpat": "wait_treatmdt_to_outpat",
}

STAGE_EVENT_PAIRS = {
    ("referral_received", "mri_performed"): "ref_to_mri",
    ("mri_performed", "mri_report_ready"): "mri_to_report",
    ("mri_report_ready", "MDT_occured"): "report_to_biopmdt",
    ("MDT_occured", "biopsy_done"): "biopmdt_to_biopsy",
    ("biopsy_done", "Path_report_recieved"): "biopsy_to_pathrep",
    ("Path_report_recieved", "Treatment_options_MDT_occured"): "pathrep_to_treatmdt",
    ("Treatment_options_MDT_occured", "Outpatient_appointment_occured"): "treatmdt_to_outpat",
}

FULL_PATHWAY_END_EVENT = "Outpatient_appointment_occured"
