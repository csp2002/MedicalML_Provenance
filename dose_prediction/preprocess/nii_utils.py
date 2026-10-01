"""NIfTI read/write helpers (from DoseDiff, https://github.com/whisney/DoseDiff, MIT).
Arrays are (z, y, x); spacing is returned/accepted in the same (z, y, x) order."""
import numpy as np
import SimpleITK as sitk

SPLITS = ("train", "validation", "test")

# 7 organs-at-risk, 3 planning target volumes and the mask of voxels that may receive dose
OARS = ["Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Esophagus", "Larynx", "Mandible"]
TARGETS = ["PTV56", "PTV63", "PTV70"]
ALL_ROIS = OARS + TARGETS + ["possible_dose_mask"]


def nii_read(path, as_type=np.float32):
    nii = sitk.ReadImage(path)
    spacing = nii.GetSpacing()  # (x, y, z)
    volume = sitk.GetArrayFromImage(nii)  # (z, y, x)
    return volume.astype(as_type), np.array(spacing[::-1], dtype=np.float32), nii.GetOrigin(), nii.GetDirection()


def nii_write(path, volume, spacing, origin, direction, as_type=np.float32):
    img = sitk.GetImageFromArray(volume.astype(as_type))
    img.SetSpacing(tuple(float(s) for s in np.asarray(spacing, dtype=np.float64)[::-1]))
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, path)
