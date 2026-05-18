import glob
import os
from pathlib import Path
from typing import Optional, Any

import SimpleITK as sitk
import matplotlib
import numpy as np
import pandas as pd
import pydicom

from data.save_load import save
from utils import ROOT

matplotlib.use("Agg")


def assert_data_exitance():
    nsclc_path = ROOT / Path("data/raw_data/NSCLC-Radiomics")
    if not os.path.exists(nsclc_path):
        raise FileNotFoundError(
            f"LUNG1 dataset not found @ {nsclc_path}. \n\
            A copy can be obtained by registering at https://www.cancerimagingarchive.net/collection/nsclc-radiomics/. \n\
            Once obtained, please paste it into `data/raw_data/`.")

    clin_path = ROOT / Path("data/raw_data/LUNG1_clinical.csv")
    if not clin_path.exists():
        raise FileNotFoundError(f"Lung1 clinical data not found @ {clin_path}. \n\
            Please retrieve it from https://www.cancerimagingarchive.net/collection/nsclc-radiomics/ and ensure it is placed under the specified path.")


def load_masked_ct(pat: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    print()
    nsclc_path = ROOT / Path("data/raw_data/NSCLC-Radiomics")

    folder_path = nsclc_path.joinpath(pat)
    if not os.path.exists(folder_path):
        print(f"Patient folder '{pat}' does not exist in NSCLC-Radiomics.")
        return None, None

    subfolder_path = next(
        (os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path)
         if os.path.isdir(os.path.join(folder_path, subfolder)) and
         len([f for f in os.listdir(os.path.join(folder_path, subfolder)) if
              os.path.isdir(os.path.join(folder_path, subfolder, f))]) == 3),
        None
    )
    if not subfolder_path:
        print(f"Subfolders missing for patient {pat}.")
        print("Patient folder must contain a subfolder with 3 subfolders (CT-slices, RT-Struct, Segmentation).")
        return None, None

    ct_folder_path = next(
        (os.path.join(subfolder_path, subfolder) for subfolder in os.listdir(subfolder_path)
         if os.path.isdir(os.path.join(subfolder_path, subfolder)) and
         len([f for f in os.listdir(os.path.join(subfolder_path, subfolder)) if f.endswith(".dcm")]) >= 20),
        None
    )
    if not ct_folder_path:
        print(f"DICOM-files missing for patient {pat}.")
        print("CT folder should contain more than 20 DICOM-files.")
        return None, None

    seg_folder_path = next(
        (os.path.join(subfolder_path, subfolder) for subfolder in os.listdir(subfolder_path)
         if os.path.isdir(os.path.join(subfolder_path, subfolder)) and "Segmentation" in subfolder and
         len([f for f in os.listdir(os.path.join(subfolder_path, subfolder)) if f.endswith(".dcm")]) == 1),
        None
    )
    if not seg_folder_path:
        print(f"Segmentation folder missing for patient {pat}.")
        print("Segmentation folder should have 'Segmentation' in its name and only contain one DICOM-file.")
        return None, None

    folder_path_part = str(folder_path).split("/data", 1)[-1]
    print(f"Loading {folder_path_part}")

    ct_paths = sorted(glob.glob(f"{ct_folder_path}/*.dcm"))
    ct = sitk.ReadImage(ct_paths)
    ct_arr = sitk.GetArrayFromImage(ct)

    seg_path = glob.glob(f"{seg_folder_path}/*.dcm")[0]
    seg = pydicom.dcmread(seg_path)

    tumor_number = None

    for seg_item in seg.SegmentSequence:
        if seg_item.SegmentDescription in ("GTV-1", "GTV_1", "GTV1"):
            tumor_number = seg_item.SegmentNumber
            break

    if not tumor_number:
        print(f"Tumor segment missing for patient {pat}.")
        print("No Tumor segment number not found in the segmentation data.")
        return None, None

    tumor_frames = []
    for frame_index, frame in enumerate(seg.PerFrameFunctionalGroupsSequence):
        if frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber == tumor_number:
            tumor_frames.append(frame_index)

    print(f"{len(tumor_frames)} tumor segmentation frames out of {len(seg.pixel_array)} total")

    mask_arr_temp = seg.pixel_array[tumor_frames]
    mask_arr = np.flip(mask_arr_temp, axis=0).copy()

    WIN = (-600, 600)
    ct_clip = np.clip(ct_arr, *WIN)
    ct_norm = ((ct_clip - WIN[0]) / (WIN[1] - WIN[0]) * 255).astype(np.uint8)

    return ct_norm, mask_arr


def select_ct_slices(mask: np.ndarray) -> tuple[list[Any], list[int]]:
    slice_idxs = [z for z in range(mask.shape[0]) if np.any(mask[z])]
    print(f"Number of slices with tumor: {len(slice_idxs)}")

    sel_slice_idxs = []
    slice_sums = [np.sum(mask[z]) for z in range(mask.shape[0])]

    if slice_idxs:
        first_idx = max(slice_idxs, key=lambda z: slice_sums[z])
    else:
        raise ValueError("No tumor slices found in the mask.")
    sel_slice_idxs.append(first_idx)
    sel_slice_rank = 1
    sel_slice_ranks = [sel_slice_rank]

    for _ in range(2):
        next_idx = max(
            (z for z in slice_idxs if all(abs(z - idx) >= 4 for idx in sel_slice_idxs)),
            key=lambda z: slice_sums[z],
            default=None
        )
        if next_idx is not None:
            sel_slice_idxs.append(next_idx)
            sel_slice_rank += 1
            sel_slice_ranks.append(sel_slice_rank)

    print(f"Number of slices selected: {len(sel_slice_idxs)}")

    return sel_slice_idxs, sel_slice_ranks


def get_ct_surv_df() -> (tuple[np.ndarray, pd.DataFrame]):
    clin_path = ROOT / Path("data/raw_data/LUNG1_clinical.csv")
    clin_df = pd.read_csv(clin_path)

    all_slices = []
    all_surv = []

    for pat in clin_df.iloc[:, 0]:
        ct, mask = load_masked_ct(pat)

        if ct is None:
            print("-----------")
            continue

        sel_slice_idxs, sel_slice_ranks = select_ct_slices(mask)
        concatenated_slices = np.stack([ct[idx] for idx in sel_slice_idxs], axis=0)

        all_slices.append(concatenated_slices)

        durations = clin_df.loc[clin_df.iloc[:, 0] == pat, "Survival.time"].values[0]
        event_observed = clin_df.loc[clin_df.iloc[:, 0] == pat, "deadstatus.event"].values[0]
        pat_num = int(clin_df.loc[clin_df.iloc[:, 0] == pat, "PatientID"].values[0][-3:])

        for i in range(len(sel_slice_idxs)):
            all_surv.extend(
                [(durations, event_observed, pat_num, sel_slice_ranks[i])]
            )

        print("-----------")

    ct_arr = np.concatenate(all_slices, axis=0)
    surv_df = pd.DataFrame(all_surv, columns=["durations", "event_observed", "pat_num", "rank"])

    return ct_arr, surv_df


def _save_ct_surv_df(arr_ct: np.ndarray, df_surv: pd.DataFrame) -> None:
    save_path = ROOT / Path("data/raw_data/ct_surv_df.pkl")
    if save_path.exists():
        os.remove(save_path)

    save(obj={"images": arr_ct, "surv_df": df_surv}, name=save_path)
    print(f"Saved CT and survival data to {save_path}")


def main():
    assert_data_exitance()
    ct_arr, surf_df = get_ct_surv_df()
    _save_ct_surv_df(ct_arr, surf_df)


if __name__ == '__main__':
    main()
