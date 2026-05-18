import torch
from math import log
from torch import Tensor
from torch import nn


class PartialMultivariateLogRankLoss(nn.Module):
    """Implements the partial multivariate log-rank loss for survival clustering as detailed in the paper
    Unsupervised risk factor identification across cancer types and data modalities via explainable artificial intelligence (https://doi.org/10.1038/s41746-026-02663-w)
    by Ferle et al (2026).

    The loss function is designed for training models to predict soft class memberships maximizing survival heterogeneity.
    It includes a penalty term to encourage balanced group memberships.

    Args:
        penalty_weight (float): Weight for the penalty term to encourage group separation. Default: 0.1.
    """

    def __init__(self, penalty_weight=0.1):
        super(PartialMultivariateLogRankLoss, self).__init__()
        self.penalty_weight = penalty_weight

    def forward(
            self,
            scores: Tensor,
            true_durations: Tensor,
            true_observations: Tensor,
    ) -> Tensor:
        """Computes the partial multivariate log-rank loss.

        Args:
            scores (Tensor): Predicted risk scores for each group.
            true_durations (Tensor): Observed survival durations.
            true_observations (Tensor): Binary event observations (1 if event occurred, 0 if censored).

        Returns:
            Tensor: Computed loss value.
        """
        self._check_inputs(scores, true_durations, true_observations)

        rm, obs = self.group_survival_from_events(scores, true_durations, true_observations)
        n_ij, n_i, Z_j, factor = self.calc_factors(rm, obs)
        logrank_loss = self.logrank_loss(n_ij, n_i, Z_j, factor)
        penalty = self.penalty(x=scores.mean(dim=0), n_groups=scores.size(dim=1)).mean()

        return logrank_loss - self.penalty_weight * penalty

    @staticmethod
    def _check_inputs(
            scores: Tensor,
            true_durations: Tensor,
            true_observations: Tensor,
    ) -> None:
        """Validates input tensors for the loss function.

        Args:
            scores (Tensor): Predicted risk scores for each group.
            true_durations (Tensor): Observed survival durations.
            true_observations (Tensor): Binary event observations (1 if event occurred, 0 if censored).

        Raises:
            ValueError: If inputs have invalid shapes or mismatched dimensions.
        """
        if true_durations.ndim != 1 or true_observations.ndim != 1:
            raise ValueError(
                f"Durations and observations must be 1D but are {true_durations.ndim} and {true_observations.ndim}.")
        if scores.ndim != 2:
            raise ValueError(f"Scores must be 2D but scores have {scores.ndim} dimensions.")
        if not scores.sum(dim=1).allclose(other=torch.ones(1), atol=1e-3):
            raise ValueError("Scores must sum to 1 in each row.")
        if scores.size(0) != true_durations.size(0) or scores.size(0) != true_observations.size(0):
            raise ValueError(
                f"Inputs must have the same length in the first dimension but have {scores.size(0)}, {true_durations.size(0)} and {true_observations.size(0)}.")

    @staticmethod
    def group_survival_from_events(
            labels: Tensor,
            durations: Tensor,
            event_observed: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Computes risk sets and observed events for each group at each unique event time.

        Args:
            labels (Tensor): Group labels for each sample.
            durations (Tensor): Observed survival durations.
            event_observed (Tensor): Binary event observations (1 if event occurred, 0 if censored).

        Returns:
            tuple: Normalized risk sets and observed events for each group at each event time.
        """
        event_times = torch.unique(durations, sorted=True)

        # Compute a boolean mask for each event time
        event_mask = durations.unsqueeze(1) == event_times

        # Compute the number of subjects at risk in each group at each event time
        rm = (event_mask.T.unsqueeze(2) * labels.unsqueeze(0)).sum(dim=1)

        # Compute the number of observed events in each group at each event time
        obs = (event_mask.T.unsqueeze(2) * labels.unsqueeze(0) * event_observed.view((1, -1, 1))).sum(
            dim=1)

        return rm / rm.sum(), obs / rm.sum()

    @staticmethod
    def calc_factors(
            rm: Tensor,
            obs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Computes intermediate factors for the log-rank.

        Args:
            rm (Tensor): Risk sets for each group at each event time.
            obs (Tensor): Observed events for each group at each event time.

        Returns:
            tuple: Intermediate factors for the log-rank:
                - n_ij: Number of partial subjects at risk in each group at each event time.
                - n_i: Total number of subjects at risk across all groups at each event time.
                - Z_j: Deviance of observed events from expected events under the null hypothesis.
                - factor: Variance factor for the log-rank statistic.
        """
        n_ij = rm.cumsum(dim=0)
        n_ij[:-1, :] = n_ij[-1, :] - n_ij[:-1, :]
        n_ij = n_ij.roll(shifts=1, dims=0)

        d_i = obs.sum(dim=1)

        n_i = rm.sum(dim=1).cumsum(dim=0)
        n_i[:-1] = n_i[-1] - n_i[:-1]
        n_i = n_i.roll(shifts=1, dims=0)

        ev = (n_ij * (d_i / n_i).view((-1, 1))).sum(dim=0)
        Z_j = obs.sum(dim=0) - ev

        factor = ((n_i - d_i) / n_i) * (d_i / n_i ** 2) + 1e-12

        return n_ij[:-1], n_i[:-1], Z_j, factor[:-1]

    def logrank_loss(
            self,
            n_ij: Tensor,
            n_i: Tensor,
            Z_j: Tensor,
            factor: Tensor,
    ) -> Tensor:
        """Computes the log-rank loss from intermediate factors.

        Args:
            n_ij (Tensor): Number of partial subjects at risk in each group at each event time.
            n_i (Tensor): Total number of subjects at risk across all groups at each event time.
            Z_j (Tensor): Deviance of observed events from expected events.
            factor (Tensor): Variance factor for the log-rank statistic.

        Returns:
            Tensor: Computed log-rank loss.
        """
        factor = torch.sqrt(factor).view((-1, 1))
        V_ = (n_ij * factor)
        V = -(V_.unsqueeze(1) * V_.unsqueeze(2)).sum(dim=0)

        n_groups = Z_j.size(0)
        ix = torch.arange(n_groups)
        V[ix, ix] = V[ix, ix] + (V_ * n_i.view((-1, 1)) * factor).sum(dim=0)

        return self.compute_ensemble_loss(ix, n_groups, V, Z_j)

    @staticmethod
    def compute_ensemble_loss(ix, n_groups, V, Z_j):
        """Computes an ensemble loss to address singularity in the covariance matrix.

        In a normal logrank test setting this would be computed as to `Z_j[:-1] @ torch.inverse(V[:-1, :-1]) @ Z_j[:-1]`,
        where one group must be excluded (usually the last).
        This is because, due to the degrees of freedom of V, one group can be represented as a combination of the others,
        thus making V is a singular matrix, which cannot be inverted.

        This, however, results to irregular gradient flow during training and tends to make groups collapse in an n_groups > 2 setting.
        Therefore, we cycle through all groups and compute the loss as an ensemble by excluding each group in their respective cycle.
        This allows for more stable group separation.

        Args:
            ix (Tensor): Indices of groups.
            n_groups (int): Number of groups.
            V (Tensor): Covariance matrix.
            Z_j (Tensor): Deviance of observed events from expected events.

        Returns:
            Tensor: Ensemble loss value.
        """
        ensemble_loss = []
        for i in ix:
            mask = torch.ones(n_groups, dtype=torch.bool)
            mask[i] = False

            V_reduced = V[mask, :][:, mask]
            Z_j_reduced = Z_j[mask]
            ensemble_loss.append(Z_j_reduced @ V_reduced.inverse() @ Z_j_reduced)
        return torch.stack(ensemble_loss).mean()

    @staticmethod
    def penalty(x: Tensor, n_groups: int) -> Tensor:
        """Computes a penalty term to encourage even group size.

        Args:
            x (Tensor): Mean scores for each group.
            n_groups (int): Number of groups.

        Returns:
            Tensor: Penalty term for group separation.
        """
        exp = log(.5) / log(1 / n_groups)
        x = x ** exp
        return 1 / (x - x ** 2) - 4
