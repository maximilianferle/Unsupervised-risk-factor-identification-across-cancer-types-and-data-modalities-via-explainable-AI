from typing import Optional

import numpy as np
import torch
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from data.datamodules.datamodule import ImageBaseDataModule


def normalize(x: torch.Tensor):
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)


def define_mnist_surv_distributions():
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


def define_group_membership():
    torch.random.manual_seed(23)
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])[torch.randperm(9)]

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


def create_mnist_survival_data():
    params_surv = define_mnist_surv_distributions()
    digits_opt = fetch_openml('optdigits', version=1, as_frame=False)
    images_tmp, labels_tmp = digits_opt.data.reshape(-1, 8, 8), digits_opt.target.astype('int64')
    images, labels = zip(*[(img, label) for img, label in zip(images_tmp, labels_tmp) if label != 0])
    images = normalize(torch.tensor(images, dtype=torch.float32))

    groups = define_group_membership()
    group_labels = torch.tensor([groups[label - 1] for label in labels])
    survival_times = torch.zeros_like(group_labels, dtype=torch.float32)
    event_observed = torch.zeros_like(group_labels, dtype=torch.float32)

    for label in group_labels.unique():
        idx = group_labels == label
        durations_tmp, event_observed_tmp = generate_weibull_times(n=len(survival_times[idx]),
                                                                   shape=params_surv["rho"][label.item()],
                                                                   scale=params_surv["lambda"][label.item()],
                                                                   censor_scale=10000,
                                                                   max_time=4000,
                                                                   seed=43)
        survival_times[idx] = torch.tensor(durations_tmp, dtype=torch.float32)
        event_observed[idx] = torch.tensor(event_observed_tmp, dtype=torch.float32)
    return images, labels, group_labels, survival_times, event_observed


class MNISTSurvivalDataModule(ImageBaseDataModule):
    def load_data(self):
        self.images, self.labels, self.group_labels, self.survival_times, self.event_observed = create_mnist_survival_data()

    def plot_mnist_survival_data(self):
        KaplanMeierFitter().fit(self.survival_times[self.group_labels == 0],
                                self.event_observed[self.group_labels == 0]).plot_survival_function()
        KaplanMeierFitter().fit(self.survival_times[self.group_labels == 1],
                                self.event_observed[self.group_labels == 1]).plot_survival_function()
        KaplanMeierFitter().fit(self.survival_times[self.group_labels == 2],
                                self.event_observed[self.group_labels == 2]).plot_survival_function()
        plt.show()
