import os
from pathlib import Path
from typing import Literal

import pandas as pd

from utils import ROOT

_file = Path(__file__)

_features = [
    "D_LAB_cbc_hemoglobin",
    "D_LAB_chem_calcium",
    "D_LAB_chem_creatinine",
    "D_LAB_chem_ldh",
    "D_LAB_chem_albumin",
    "D_LAB_serum_beta2_microglobulin",
    "D_LAB_serum_m_protein",
    "D_LAB_serum_lambda",
    "D_LAB_serum_kappa",
    "D_LAB_cbc_wbc",
]

_rename_feat = (
    _features,
    [
        "Hemoglobin",
        "Calcium",
        "Creatinine",
        "LDH",
        "Albumin",
        fr"$\beta$-2-Microglobulin",
        "M-Protein",
        fr"SFL-$\lambda$",
        fr"SFL-$\kappa$",
        "WBC",
    ],
)
_units = [
    "mmol/L",
    "mmol/L",
    r"$\mu$mol/L",
    r"$\mu$kat/L",
    "g/L",
    "mg/L",
    "g/dL",
    "mg/dL",
    "mg/dL",
    r"10$^{-9}$ cells/L",
]
_survival = ["deathdy",
             "lstalive",
             "lastdy1",
             "pddy1",
             "pdflag1", ]

_pid = "PUBLIC_ID"
_visit = "VISIT"


def _data_lookup(file_path: Path | str):
    """Validates the existence of a dataset file.

    Args:
        file_path (Path | str): Path to the dataset file.

    Raises:
        FileNotFoundError: If the file does not exist, with instructions for obtaining the dataset.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"CoMMpass dataset not found @ {file_path}. \n\
            A copy can be obtained by registering at https://research.themmrf.org/. \n\
            For this project, the clinical flatfiles `MMRF_CoMMpass_IA21_PER_PATIENT_VISIT.tsv` & `MMRF_CoMMpass_IA21_STAND_ALONE_SURVIVAL.tsv` were used. \n\
            Once obtained, please paste it into `data/raw_data/` or indicate its path when calling this function.")


def _read_commpass_df(file_path: Path | str = None) -> pd.DataFrame:
    """Reads the CoMMpass dataset from a TSV file.

    Args:
        file_path (Path | str, optional): Path to the dataset file. If None, a default path is used.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    if not file_path:
        file_path = ROOT / "data/raw_data/MMRF_CoMMpass_IA21_PER_PATIENT_VISIT.tsv"
        _data_lookup(file_path=file_path)

    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    return df


def _prepare_feature_df(file_path: Path | str = None) -> pd.DataFrame:
    """Prepares a feature DataFrame from the CoMMpass dataset.

    Filters the dataset for the initial visit (VISIT == 0), selects specific features,
    and drops rows with missing values.

    Args:
        file_path (Path | str, optional): Path to the dataset file. If None, a default path is used.

    Returns:
        pd.DataFrame: Processed feature DataFrame.
    """
    df = _read_commpass_df(file_path=file_path)
    df = df[(df[_visit] == 0)]
    df = df[[_pid,
             _visit,
             *_features]]
    df.dropna(inplace=True)

    return df


def _prepare_nan_containing_feature_df(file_path: Path | str = None) -> pd.DataFrame:
    """Prepares a feature DataFrame with imputation for missing values.

    Imputes missing values using backward filling within each patient group,
    filters for the initial visit (VISIT == 0), and removes duplicate patients.

    Args:
        file_path (Path | str, optional): Path to the dataset file. If None, a default path is used.

    Returns:
        pd.DataFrame: Processed feature DataFrame with imputed values.
    """
    df = _read_commpass_df(file_path=file_path)

    df = df[[_pid,
             _visit,
             *_features]]
    df = df.groupby(_pid).apply(lambda group: group.bfill())
    df = df[(df[_visit] == 0)]
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(subset=_pid, inplace=True)
    return df


def _prepare_survival_df(file_path: Path | str = None, survival: Literal["pfs", "os"] = "os") \
        -> pd.DataFrame:
    """Prepares a survival DataFrame for either Progression-Free Survival (PFS) or Overall Survival (OS).

    Processes survival-related columns to compute `durations` and `event_observed` based on the chosen survival type.

    Args:
        file_path (Path | str, optional): Path to the survival dataset file. If None, a default path is used.
        survival (Literal["pfs", "os"]): Type of survival analysis. "pfs" for Progression-Free Survival,
            "os" for Overall Survival. Default: "os".

    Returns:
        pd.DataFrame: Processed survival DataFrame with `durations` and `event_observed` columns.
    """
    if not file_path:
        file_path = ROOT / "data/raw_data/MMRF_CoMMpass_IA21_STAND_ALONE_SURVIVAL.tsv"
        _data_lookup(file_path=file_path)

    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    df = df[[_pid,
             *_survival]]
    df.dropna(subset=_survival, how="all", inplace=True)

    if survival == "pfs":
        df["durations"] = df["pddy1"].fillna(df["lastdy1"])
        df["event_observed"] = df["pdflag1"]
        df.drop(columns=_survival, inplace=True)

    elif survival == "os":
        df["durations"] = df["deathdy"].fillna(df["lstalive"])
        df["event_observed"] = df["deathdy"].notna().astype(int)
        df.drop(columns=_survival, inplace=True)

    return df.dropna()


def _harmonize_dfs(df1: pd.DataFrame, df2: pd.DataFrame, on: str, how: str) -> (pd.DataFrame, pd.DataFrame):
    """Merges feature and survival DataFrames and splits them into harmonized DataFrames.

    Args:
        df1 (pd.DataFrame): Feature DataFrame.
        df2 (pd.DataFrame): Survival DataFrame.
        on (str): Column name to merge on.
        how (str): Type of merge to perform.

    Returns:
        tuple: A tuple containing:
            - df_feat (pd.DataFrame): Harmonized feature DataFrame.
            - df_surv (pd.DataFrame): Harmonized survival DataFrame.
    """
    df_temp = pd.merge(df1, df2, on=on, how=how)
    df_temp.set_index("PUBLIC_ID", inplace=True)
    df_feat = df_temp[_features]
    df_surv = df_temp[["durations", "event_observed"]]

    return df_feat, df_surv


def get_nan_containing_feat_surv_df(file_path: Path | str = None, survival: Literal["pfs", "os"] = "os") -> (
        pd.DataFrame, pd.DataFrame):
    """Returns harmonized feature and survival DataFrames with imputed missing values.

    Args:
        file_path (Path | str, optional): Path to the dataset file. If None, a default path is used.
        survival (Literal["pfs", "os"]): Type of survival analysis. "pfs" for Progression-Free Survival,
            "os" for Overall Survival. Default: "os".

    Returns:
        tuple: A tuple containing:
            - df_feat (pd.DataFrame): Feature DataFrame with imputed missing values.
            - df_surv (pd.DataFrame): Survival DataFrame.
    """
    df_feat_tmp = _prepare_nan_containing_feature_df(file_path=file_path)
    df_surv_tmp = _prepare_survival_df(file_path=file_path, survival=survival)

    return _harmonize_dfs(df_feat_tmp, df_surv_tmp, on=_pid, how="inner")


def get_feat_surv_df(file_path: Path | str = None) \
        -> (pd.DataFrame, pd.DataFrame):
    """Returns harmonized feature and survival DataFrames without missing values.

    Args:
        file_path (Path | str, optional): Path to the dataset file. If None, a default path is used.

    Returns:
        tuple: A tuple containing:
            - df_feat (pd.DataFrame): Feature DataFrame without missing values.
            - df_surv (pd.DataFrame): Survival DataFrame.
    """
    df_feat_tmp = _prepare_feature_df(file_path=file_path)
    df_surv_tmp = _prepare_survival_df(file_path=file_path)

    return _harmonize_dfs(df_feat_tmp, df_surv_tmp, on=_pid, how="inner")
