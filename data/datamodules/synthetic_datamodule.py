from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter

from data.datamodules.datamodule import BaseDataModule


def define_mlp_surv_distributions():
    params = {
        "rho": {
            0: 0.539,
            1: 0.898,
            2: 1.257,
        },
        "lambda": {
            0: 3068.812,
            1: 5114.687,
            2: 7160.562,
        }
    }

    return params


def define_mlp_feature_distributions():
    params = {
        "mu": {
            0: [1, 0, 0],
            1: [0, 1, 0],
            2: [0, 0, 1],
        },
        "sigma": {
            0: [0.5, 0.5, 0.5],
            1: [0.5, 0.5, 0.5],
            2: [0.5, 0.5, 0.5],
        }
    }

    return params


def define_group_membership():
    torch.random.manual_seed(23)
    groups = torch.tensor([0, 1, 2])[torch.randperm(3)]

    return groups


def generate_weibull_times(
        n: int,
        shape: float,
        scale: float,
        censor_scale: Optional[float] = None,
        max_time: Optional[float] = None,
        seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Inverse‑transform sampling for Weibull(k, λ)
    u = rng.uniform(size=n)
    event_times = scale * (-np.log1p(-u)) ** (1 / shape)

    if censor_scale is not None:
        censor_times = rng.exponential(scale=censor_scale, size=n)
        durations = np.minimum(event_times, censor_times)
        event_observed = event_times <= censor_times
    else:
        durations = event_times.copy()
        event_observed = np.ones_like(event_times, dtype=bool)

    # Apply administrative cut‑off if requested
    if max_time is not None:
        durations = np.minimum(durations, max_time)
        # An event is observed only if it happened *before* the admin cut‑off
        event_observed = event_observed & (event_times <= durations)

    return durations, event_observed


def create_mlp_survival_data():
    torch.random.manual_seed(43)
    params_surv = define_mlp_surv_distributions()
    params_feat = define_mlp_feature_distributions()

    groups = define_group_membership()
    group_labels = torch.cat([groups[0].repeat(500), groups[1].repeat(500), groups[2].repeat(500)])
    features = torch.zeros((len(group_labels), len(groups)), dtype=torch.float32)
    survival_times = torch.zeros_like(group_labels, dtype=torch.float32)
    event_observed = torch.zeros_like(group_labels, dtype=torch.float32)

    for label in group_labels.unique():
        idx = group_labels == label
        for feat in range(len(params_feat["mu"][label.item()])):
            features[idx, feat] = torch.distributions.Normal(loc=params_feat["mu"][label.item()][feat],
                                                             scale=params_feat["sigma"][label.item()][feat]).sample(
                sample_shape=[idx.sum().item(), ])

        durations_tmp, event_observed_tmp = generate_weibull_times(n=len(survival_times[idx]),
                                                                   shape=params_surv["rho"][label.item()],
                                                                   scale=params_surv["lambda"][label.item()],
                                                                   censor_scale=10000,
                                                                   max_time=4000,
                                                                   seed=43)

        survival_times[idx] = torch.tensor(durations_tmp, dtype=torch.float32)
        event_observed[idx] = torch.tensor(event_observed_tmp, dtype=torch.float32)

    idx = torch.randperm(1500)
    features_final, durations_final, event_observed_final, group_labels_final = features[idx], survival_times[idx], \
    event_observed[idx], group_labels[idx]

    df_feat = pd.DataFrame(features_final.numpy(), columns=[str(i) for i in range(features_final.shape[1])])
    df_surv = pd.DataFrame({
        "durations": durations_final.numpy(),
        "event_observed": event_observed_final.numpy()
    })

    return df_feat, df_surv, group_labels_final


class SyntheticDataModule(BaseDataModule):
    def _init(self,
              batch_size=64,
              train_val_split=0.8, ):
        self.load_data()
        self.input_size = self.df_feat.shape[1]
        self.batch_size = batch_size

        self.idx_train, self.idx_val = self._get_split_idxs(split_ratio=train_val_split)
        self._train, self._val, self._test = self._split_data(self.idx_train, self.idx_val)
        self._standard_scaler = self._fit_standard_scaler()

    def load_data(self):
        self.df_feat, self.df_surv, self.group_labels = create_mlp_survival_data()

    def plot_mnist_survival_data(self):
        survival_times = self.df_surv["durations"].to_numpy()
        event_observed = self.df_surv["event_observed"].to_numpy()

        for group in self.group_labels.unique():
            group = int(group)

            kmf = KaplanMeierFitter()
            kmf.fit(survival_times[self.group_labels == group],
                    event_observed[self.group_labels == group],
                    label=f'Group {group}')
            kmf.plot_survival_function(show_censors=True)

        plt.show()
