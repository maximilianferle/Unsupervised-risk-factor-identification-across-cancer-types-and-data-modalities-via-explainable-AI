import os

import torch

from data.datamodules.datamodule import ImageBaseDataModule
from data.image_normalization import normalize
from data.save_load import load
from utils import ROOT


def prepare_lung_survival_data(pkl_file_path: str = "data/raw_data/ct_surv_df.pkl"):
    if not os.path.exists(pkl_file_path):
        from data import dcm_preprocessing
        dcm_preprocessing.main()
    data = load(ROOT / pkl_file_path)
    images, surv_df = data["images"], data["surv_df"]

    images = normalize(images)
    images = torch.tensor(images, dtype=torch.float32)

    survival_times_tmp = surv_df["durations"].values
    survival_times = torch.tensor(survival_times_tmp, dtype=torch.float32)

    event_observed_tmp = surv_df["event_observed"].values
    event_observed = torch.tensor(event_observed_tmp, dtype=torch.float32)

    pat_num_tmp = surv_df["pat_num"].values
    pat_num = torch.tensor(pat_num_tmp, dtype=torch.float32)

    rank_tmp = surv_df["rank"].values
    rank = torch.tensor(rank_tmp, dtype=torch.float32)

    return images, survival_times, event_observed, pat_num, rank


class LUNGSurvivalDataModule(ImageBaseDataModule):
    def load_data(self):
        self.images, self.survival_times, self.event_observed, self.pat_num, self.rank = prepare_lung_survival_data()

    def _get_split_idxs(self, split_ratio: float) -> tuple[list[int], list[int]]:
        torch.random.manual_seed(self.seed)

        assert 0 <= split_ratio <= 1, 'split_ratio must be between 0 and 1idx_val'
        split_ratio = round(min(split_ratio, 1 - split_ratio), ndigits=1)
        k = round(min(self.k, 1 / split_ratio))

        pat_num_unique = torch.unique(self.pat_num)
        n_samples = len(pat_num_unique)
        n_val_samples = int(n_samples * split_ratio)
        pat_num_rand = pat_num_unique[torch.randperm(len(pat_num_unique))]

        pat_val = pat_num_rand[(k - 1) * n_val_samples:k * n_val_samples]
        uniques, counts = torch.cat((pat_val, pat_num_rand)).unique(return_counts=True)
        pat_train = uniques[counts == 1]

        # idx_val = [i for i, pat in enumerate(self.pat_num) if pat in pat_val and self.rank[i] == 1]
        idx_val = [i for i, pat in enumerate(self.pat_num) if pat in pat_val]
        idx_train = [i for i, pat in enumerate(self.pat_num) if pat in pat_train]

        return idx_train, idx_val
