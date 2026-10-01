"""OpenKBP evaluation: dose score and DVH score.

Dose score: mean absolute error (Gy) between predicted and reference dose inside the
            possible-dose mask, averaged over patients.
DVH score:  mean absolute error (Gy) of the DVH metrics over all patients and structures:
            targets (PTV56/63/70): D1, D95, D99; organs at risk: D0.1cc and Dmean.

Used by dosediff/test.py and c3d/test.py.
(Adapted by DoseDiff from https://github.com/ababier/open-kbp, MIT; see LICENSE_OPENKBP.)
"""
import os

import numpy as np
import SimpleITK as sitk

STRUCTURES = ["Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Esophagus", "Larynx", "Mandible",
              "PTV70", "PTV63", "PTV56"]


def dose_mae(pred, gt, possible_dose_mask):
    return float(np.mean(np.abs(pred[possible_dose_mask > 0] - gt[possible_dose_mask > 0])))


def dvh_metrics(dose, mask, is_target, spacing):
    roi_dose = dose[mask > 0]
    if is_target:
        return {"D1": np.percentile(roi_dose, 99), "D95": np.percentile(roi_dose, 5), "D99": np.percentile(roi_dose, 1)}
    voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / np.prod(spacing)))
    return {"D_0.1_cc": np.percentile(roi_dose, 100 - voxels_in_tenth_of_cc / len(roi_dose) * 100),
            "mean": np.mean(roi_dose)}


def read(path, pixel_type=None):
    img = sitk.ReadImage(path, pixel_type) if pixel_type is not None else sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img), img.GetSpacing()


def evaluate(prediction_dir, gt_dir, patients=None):
    """prediction_dir/<pt>/dose.nii.gz vs gt_dir/<pt>/{dose,possible_dose_mask,<structure>}.nii.gz, for the given
    patients (default: every patient in prediction_dir). Returns (summary dict, per-patient rows)."""
    dose_errors, dvh_errors, rows = [], [], []
    for patient in (patients or sorted(os.listdir(prediction_dir))):
        pred, _ = read(os.path.join(prediction_dir, patient, "dose.nii.gz"))
        gt, _ = read(os.path.join(gt_dir, patient, "dose.nii.gz"))
        body, _ = read(os.path.join(gt_dir, patient, "possible_dose_mask.nii.gz"))
        mae = dose_mae(pred, gt, body)
        dose_errors.append(mae)
        patient_dvh = []
        for s in STRUCTURES:
            path = os.path.join(gt_dir, patient, f"{s}.nii.gz")
            if not os.path.exists(path):  # not every patient has every structure
                continue
            mask, spacing = read(path, sitk.sitkUInt8)
            is_target = s.startswith("PTV")
            ref, new = dvh_metrics(gt, mask, is_target, spacing), dvh_metrics(pred, mask, is_target, spacing)
            patient_dvh += [abs(ref[k] - new[k]) for k in ref]
        dvh_errors += patient_dvh
        rows.append(dict(patient=patient, dose_mae=mae, dvh_mae=float(np.mean(patient_dvh))))
    summary = dict(dose_score=float(np.mean(dose_errors)), dose_score_std=float(np.std(dose_errors)),
                   dvh_score=float(np.mean(dvh_errors)), dvh_score_std=float(np.std(dvh_errors)),
                   num_patients=len(rows))
    return summary, rows
