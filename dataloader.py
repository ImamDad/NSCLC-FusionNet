"""
dataloader.py
--------------
Dataset and path configuration for NSCLC-FusionNet.

This file is the single source of truth for:
  - where every dataset lives on disk,
  - what folder structure each dataset is expected to follow,
  - how a patient is assembled from its modalities.

To run on your own data, edit the PATHS dict at the top of this file.
Every other part of the repository reads paths from here.

Each patient is represented as a dict with keys:
    "wsi"       : torch.Tensor (N_patches, 1024)  or None
    "ct"        : torch.Tensor (1, 128, 128, 128) or None
    "pet"       : torch.Tensor (1, 128, 128, 128) or None
    "rna"       : torch.Tensor (50,)               or None
    "clin"      : torch.Tensor (n_clin,)            or None
    "labels"    : dict of task -> int               or None
    "survival"  : (time, event)                     or None
    "mask"      : torch.Tensor (5,)  1=observed, 0=missing
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

# ======================================================================
# 1. EDIT THESE PATHS
# ======================================================================
# Base directory that holds all datasets.
DATA_ROOT = Path(os.environ.get("NSCLC_DATA_ROOT", "./data"))

# Per-dataset paths. Each key points to the folder that contains the
# modality subfolders and the clinical.csv for that cohort.
PATHS = {
    # ---- Phase 2: development cohorts -------------------------------
    "tcga_luad": {
        "root":      DATA_ROOT / "tcga_luad",
        "wsi_dir":   DATA_ROOT / "tcga_luad" / "wsi",      # <patient_id>.pt  (pre-extracted UNI features)
        "rna_dir":   DATA_ROOT / "tcga_luad" / "rna",      # <patient_id>.npy (50 GSVA Hallmark scores)
        "clinical":  DATA_ROOT / "tcga_luad" / "clinical.csv",
    },
    "tcga_lusc": {
        "root":      DATA_ROOT / "tcga_lusc",
        "wsi_dir":   DATA_ROOT / "tcga_lusc" / "wsi",
        "rna_dir":   DATA_ROOT / "tcga_lusc" / "rna",
        "clinical":  DATA_ROOT / "tcga_lusc" / "clinical.csv",
    },
    "radiogenomics": {
        "root":      DATA_ROOT / "radiogenomics",
        "ct_dir":    DATA_ROOT / "radiogenomics" / "ct",   # <patient_id>.npy (128^3)
        "pet_dir":   DATA_ROOT / "radiogenomics" / "pet",  # <patient_id>.npy (128^3)
        "rna_dir":   DATA_ROOT / "radiogenomics" / "rna",
        "clinical":  DATA_ROOT / "radiogenomics" / "clinical.csv",
    },
    # ---- Phase 3: external cohorts ----------------------------------
    "cptac": {
        "root":      DATA_ROOT / "cptac",
        "wsi_dir":   DATA_ROOT / "cptac" / "wsi",
        "rna_dir":   DATA_ROOT / "cptac" / "rna",
        "clinical":  DATA_ROOT / "cptac" / "clinical.csv",
    },
    "lung1": {
        "root":      DATA_ROOT / "lung1",
        "ct_dir":    DATA_ROOT / "lung1" / "ct",
        "clinical":  DATA_ROOT / "lung1" / "clinical.csv",
    },
    "acrin6668": {
        "root":      DATA_ROOT / "acrin6668",
        "pet_dir":   DATA_ROOT / "acrin6668" / "pet",
        "ct_dir":    DATA_ROOT / "acrin6668" / "ct",
        "clinical":  DATA_ROOT / "acrin6668" / "clinical.csv",
    },
}

# Column names expected in each clinical.csv.
CLINICAL_COLUMNS = {
    "patient_id": "patient_id",
    "age":        "age",
    "sex":        "sex",          # 0/1
    "smoking":    "smoking",      # 0/1/2
    "stage":      "stage_group",  # I-IV -> 0..3
    "subtype":    "histology",    # 0=LUAD, 1=LUSC
    "T":          "t_category",   # 0..3
    "N":          "n_status",     # 0= N0, 1= N+
    "driver":     "driver",       # 0=EGFR, 1=KRAS, 2=neither
    "os_time":    "os_time",
    "os_event":   "os_event",
    "rec_time":   "rec_time",     # optional
    "rec_event":  "rec_event",    # optional
}

# Number of clinical features the model expects (age, sex, smoking, ...)
N_CLIN = 5

# ======================================================================
# 2. DATASET
# ======================================================================
class NSCLCFusionDataset(Dataset):
    """
    Loads one cohort and returns a per-patient dict of modality tensors.
    Missing modalities are returned as None, and the mask reflects which
    modalities are observed.
    """

    MODALITIES = ["wsi", "ct", "pet", "rna", "clin"]

    def __init__(self, cohort: str, transform=None):
        assert cohort in PATHS, f"Unknown cohort {cohort}. Choose from {list(PATHS)}"
        self.cohort = cohort
        self.paths = PATHS[cohort]
        self.transform = transform
        self.df = pd.read_csv(self.paths["clinical"])
        self.patient_ids = self.df[CLINICAL_COLUMNS["patient_id"]].astype(str).tolist()

    def __len__(self):
        return len(self.patient_ids)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _load_wsi(self, pid):
        p = self.paths.get("wsi_dir", None)
        if p is None:
            return None
        f = os.path.join(p, f"{pid}.pt")
        if not os.path.exists(f):
            return None
        return torch.load(f, map_location="cpu")  # (N_patches, 1024)

    def _load_volume(self, pid, key):
        p = self.paths.get(f"{key}_dir", None)
        if p is None:
            return None
        f = os.path.join(p, f"{pid}.npy")
        if not os.path.exists(f):
            return None
        arr = np.load(f).astype(np.float32)       # (128,128,128)
        return torch.from_numpy(arr).unsqueeze(0)  # (1,128,128,128)

    def _load_rna(self, pid):
        p = self.paths.get("rna_dir", None)
        if p is None:
            return None
        f = os.path.join(p, f"{pid}.npy")
        if not os.path.exists(f):
            return None
        return torch.from_numpy(np.load(f).astype(np.float32))  # (50,)

    def _load_clin(self, row):
        c = CLINICAL_COLUMNS
        vals = [
            float(row.get(c["age"], 0.0)) / 100.0,
            float(row.get(c["sex"], 0.0)),
            float(row.get(c["smoking"], 0.0)) / 2.0,
            float(row.get(c["stage"], 0.0)) / 3.0,
            1.0,   # treatment indicator; replace with real column if available
        ]
        return torch.tensor(vals, dtype=torch.float32)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row[CLINICAL_COLUMNS["patient_id"]])

        sample = {
            "patient_id": pid,
            "cohort":     self.cohort,
            "wsi":        self._load_wsi(pid),
            "ct":         self._load_volume(pid, "ct"),
            "pet":        self._load_volume(pid, "pet"),
            "rna":        self._load_rna(pid),
            "clin":       self._load_clin(row),
        }

        # availability mask
        mask = torch.tensor(
            [1.0 if sample[m] is not None else 0.0 for m in self.MODALITIES],
            dtype=torch.float32,
        )
        sample["mask"] = mask

        # labels (some cohorts may not have all tasks)
        c = CLINICAL_COLUMNS
        sample["labels"] = {
            "subtype":    int(row[c["subtype"]]) if c["subtype"] in row else None,
            "T":          int(row[c["T"]])       if c["T"]       in row else None,
            "N":          int(row[c["N"]])       if c["N"]       in row else None,
            "driver":     int(row[c["driver"]])  if c["driver"]  in row else None,
            "stage":      int(row[c["stage"]])   if c["stage"]   in row else None,
        }
        sample["survival"] = (
            float(row[c["os_time"]]),
            float(row[c["os_event"]]),
        )
        if c["rec_time"] in row and c["rec_event"] in row:
            sample["recurrence"] = (
                float(row[c["rec_time"]]),
                float(row[c["rec_event"]]),
            )
        else:
            sample["recurrence"] = None

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ======================================================================
# 3. COLLATE
# ======================================================================
def collate_fn(batch):
    """
    Custom collate that keeps modality tensors as lists (because WSI has
    variable N_patches) and stacks everything else.
    """
    out = {
        "patient_id": [b["patient_id"] for b in batch],
        "cohort":     [b["cohort"] for b in batch],
        "wsi":        [b["wsi"] for b in batch],
        "ct":         [b["ct"] for b in batch],
        "pet":        [b["pet"] for b in batch],
        "rna":        [b["rna"] for b in batch],
        "clin":       torch.stack([b["clin"] for b in batch]),
        "mask":       torch.stack([b["mask"] for b in batch]),
        "labels":     [b["labels"] for b in batch],
        "survival":   torch.tensor([b["survival"] for b in batch], dtype=torch.float32),
        "recurrence": [b["recurrence"] for b in batch],
    }
    return out


# ======================================================================
# 4. SMOKE TEST
# ======================================================================
if __name__ == "__main__":
    for cohort in PATHS:
        try:
            ds = NSCLCFusionDataset(cohort)
            print(f"{cohort:15s}  {len(ds):5d} patients  "
                  f"clinical.csv = {PATHS[cohort]['clinical']}")
        except FileNotFoundError:
            print(f"{cohort:15s}  clinical.csv not found (edit PATHS)")
