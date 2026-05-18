from typing import Literal
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from data.datamodules.datamodule import BaseDataModule
from net.models import CudaLungConvNetwork
from stats.weighted_concordance import weighted_concordance_index
from utils import ROOT, assert_partition



def plot_single_net_prediction(
        v_num: int,
        model=None,
        dm: Type[BaseDataModule] = None,
        partition: Literal["val_dataset", "test_dataset"] = "val_dataset",
        **kwargs,
):
    assert_partition(partition)
    assert dm is not None

    if not model:
        from net.training import TrainableNetwork
        checkpoint_path = max((ROOT / f'net/models/lightning_logs/version_{v_num}/checkpoints').glob('*.ckpt'),
                              key=lambda p: p.stat().st_mtime)
        model = TrainableNetwork.load_from_checkpoint(checkpoint_path).model
    if not hasattr(dm, partition):
        dm.setup()

    features, (durations, event_observed) = getattr(dm, partition)[:]

    y_hat = model(features)

    order = rank_classes_by_mean_survival(classes=y_hat.argmax(dim=1), durations=durations,
                                          event_observed=event_observed)

    ax = make_prediction_plot(
        predictions=y_hat[:, order],
        durations=durations,
        event_observed=event_observed,
        weights=torch.ones_like(durations),
        **kwargs)

    fig_path = f'figures/plots/{dm.__class__.__name__.replace("DataModule", "")}_v{v_num}_{partition}.png'
    plt.savefig(ROOT / fig_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
    print(f"Prediction successfully saved @ {fig_path}.")
    plt.clf()


def get_val_weights(dm: Type[BaseDataModule]):
    val_pat_num = dm.pat_num[dm.idx_val]
    mask = val_pat_num[:, None] == torch.unique(val_pat_num)
    weights = (mask * 1 / mask.sum(0, keepdim=True)).sum(dim=1)
    return weights


def _get_kaplan_meier_fitters(classes, durations, event_observed, weights=None):
    fitters = [KaplanMeierFitter().fit(
        durations=durations[classes == c],
        event_observed=event_observed[classes == c],
        weights=weights[classes == c] if weights is not None else torch.ones_like(
            durations[classes == c]),
    ) for c in classes.unique()]
    return fitters


def rank_classes_by_mean_survival(classes, durations, event_observed, weights=None):
    fitters = _get_kaplan_meier_fitters(classes=classes, durations=durations, event_observed=event_observed,
                                        weights=weights)
    fractions = [calc_expected_value(fitter) for fitter in fitters]
    order = np.argsort(fractions)
    return order


def calc_expected_value(fitter: KaplanMeierFitter):
    survival_function = fitter.survival_function_.to_numpy().squeeze()
    timeline = fitter.timeline
    expected_value = np.sum((survival_function * timeline)) / np.sum(survival_function)
    return expected_value


def get_combined_model_output(
        dm_cls: Type[BaseDataModule],
        n_classes: int = 2,
        v_nums: list | range = range(5),
        lightning_logs_path: str = "net/models/lightning_logs/",
        calc_weights: bool = False,
        partition: Literal["val_dataset", "test_dataset"] = "val_dataset",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert_partition(partition)
    from net.training import TrainableNetwork

    predictions = torch.empty((0, n_classes))
    all_durations = torch.empty(0)
    all_event_observed = torch.empty(0)
    all_weights = torch.empty(0)
    all_features = torch.empty((0, 1))

    for k, v_num in enumerate(v_nums):
        checkpoint_path = max((ROOT / lightning_logs_path / f'version_{v_num}/checkpoints').glob('*.ckpt'),
                              key=lambda p: p.stat().st_mtime)
        model = TrainableNetwork.load_from_checkpoint(checkpoint_path, weights_only=False).model.cpu()
        dm = dm_cls(k=k + 1, use_cached=False)
        dm.setup()

        features, (durations, event_observed) = getattr(dm, partition)[:]
        all_features = all_features.reshape(-1, *features.shape[1:])
        all_features = torch.cat(
            (all_features,
             torch.from_numpy(dm._standard_scaler.inverse_transform(features)) if hasattr(dm,
                                                                                          "_standard_scaler") else features)
            , dim=0)

        weights = get_val_weights(dm=dm) if calc_weights else torch.ones_like(durations)

        y_hat = model(features)
        classes = y_hat.argmax(dim=1)
        order = rank_classes_by_mean_survival(classes, durations, event_observed, weights)
        y_hat = y_hat[:, order]

        predictions = torch.cat((predictions, y_hat), dim=0)
        all_durations = torch.cat((all_durations, durations), dim=0)
        all_event_observed = torch.cat((all_event_observed, event_observed), dim=0)
        all_weights = torch.cat((all_weights, weights), dim=0)

    return predictions, all_durations, all_event_observed, all_weights, all_features


def make_prediction_plot(
        predictions: torch.Tensor,
        durations: torch.Tensor,
        event_observed: torch.Tensor,
        weights: torch.Tensor,
        ylabel: str = "Progression-free Survival [a.u.]",
        xlim: tuple = (None, None),
        time_scale: Literal["days", "months"] = "days",
        **kwargs,
):
    if predictions.ndim == 2:
        classes = predictions.argmax(dim=1)
    else:
        classes = predictions
    fig, ax = plt.subplots(1, 1, figsize=(8, 5 * (4 / 3)))
    for c in classes.unique():
        mask = classes == c
        kmf = KaplanMeierFitter().fit(
            durations=durations[mask],
            event_observed=event_observed[mask],
            weights=weights[mask])
        print(kmf.median_survival_time_)
        kmf.plot_survival_function(ax=ax,
                                   label=f"Cluster {c}",
                                   show_censors=True,
                                   **kwargs)

    p = multivariate_logrank_test(
        event_durations=durations, groups=classes,
        event_observed=event_observed,
        weights=weights.numpy()).p_value
    p = f"{p:.1e}".replace("e", fr"~*~10^{{") + "}"
    c_idx = 1 - weighted_concordance_index(event_times=durations.detach().numpy(),
                                           event_observed=event_observed.detach().numpy(),
                                           predicted_scores=classes.detach().numpy(),
                                           weights=weights.detach().numpy())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("time [years]")
    ax.set_xlim(*xlim)
    t = 365.25 if time_scale == "days" else 12 if time_scale == "months" else 1
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x / t:.1f}"))
    ax.set_ylabel(ylabel)
    plt.text(0.9, 0.9, fr"$p~=~{p}$", transform=ax.transAxes, fontsize=9, ha='right', va='center')
    plt.text(0.9, 0.8, fr"$c~=~{round(c_idx, 3)}$", transform=ax.transAxes, fontsize=9, ha='right', va='center')
    return ax


def save_plot(ax: plt.Axes, fig_path: str):
    plt.sca(ax)
    plt.savefig(ROOT / fig_path,
                bbox_inches='tight', transparent=True, dpi=300)
    print(f"Prediction successfully saved @ {fig_path}.")


def plot_all_net_predictions(
        dm_cls: Type[BaseDataModule],
        n_classes: int = 2,
        v_nums: list | range = range(5),
        lightning_logs_path: str = "lightning_logs/",
        calc_weights: bool = False,
        partition: Literal["val_dataset", "test_dataset"] = "val_dataset",
        **kwargs,
):
    assert_partition(partition)
    predictions, all_durations, all_event_observed, all_weights, *_ = get_combined_model_output(
        dm_cls=dm_cls,
        n_classes=n_classes,
        v_nums=v_nums,
        lightning_logs_path=lightning_logs_path,
        calc_weights=calc_weights,
        partition=partition
    )
    if partition == "test_dataset":
        predictions, _ = predictions.argmax(dim=-1).reshape((5, -1)).mode(dim=0)
        all_durations = all_durations.reshape((5, -1))[0, :]
        all_event_observed = all_event_observed.reshape((5, -1))[0, :]
        all_weights = all_weights.reshape((5, -1))[0, :]

    ax = make_prediction_plot(
        predictions=predictions,
        durations=all_durations,
        event_observed=all_event_observed,
        weights=all_weights,
        **kwargs)

    fig_path = f'figures/plots/{dm_cls.__name__.replace("DataModule", "")}_survival_all_{partition}.png'
    save_plot(ax=ax, fig_path=fig_path)
