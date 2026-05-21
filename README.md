# Unsupervised Risk Factor Identification via Explainable Artificial Intelligence

This repository contains the accompanying code for the paper:
**[Unsupervised risk factor identification across cancer types and data modalities via explainable artificial intelligence](https://doi.org/10.1038/s41746-026-02663-w)** by Ferle et al (2026).

## Overview

This repository provides implementations of survival models using the **partial multivariate log-rank loss** for
survival-guided clustering. It includes:

- Data preprocessing pipelines for clinical and synthetic datasets.
- Deep learning models for survival analysis.
- The `PartialMultivariateLogRankLoss` loss module, as the main innovation of this work.

## Reproducing Results

To reproduce the core results of the study, install dependencies via

```bash
pip install -r requirements.txt
```

and execute:

```bash
python main.py
```

### Notes on Data

- **Synthetic data experiments** can be reproduced as-is.
- **CoMMpass and Lung1 datasets** are **not included** in this repository. Users must obtain their own copies of these
  datasets from the respective sources:
    - CoMMpass: [https://research.themmrf.org/](https://research.themmrf.org/)
    - Lung1: [https://www.cancerimagingarchive.net/collection/nsclc-radiomics/](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/)

## Core Innovation: Partial Multivariate Log-Rank Loss

The main innovation of this work is implemented in `loss/partial_multivariate_logrank.py`. This loss module enables
model architecture- and data modality-independent clustering of risk groups in a survival-guided manner while handling
censored data.

### Example Usage

```python
import torch
import torch.nn as nn

from loss.partial_multivariate_logrank import PartialMultivariateLogRankLoss


class MinimalSurvivalModel(nn.Module):
    """A minimal neural network"""

    def __init__(self, input_dim, n_groups=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_groups)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """The model should return soft group assignments, which sum to 1 for each sample (equivalently to a classifier)"""
        x = self.fc(x)
        return self.softmax(x)


def train_model(
        batch_size=10,
        input_dim=5,
        n_groups=3,
        n_epochs=20,
):
    """A minimal training loop for a survival model."""
    x = torch.rand(batch_size, input_dim)  # Input features
    true_durations = torch.rand(batch_size)  # Observed survival durations
    event_observed = torch.randint(0, 2, (batch_size,))  # Binary event observations

    model = MinimalSurvivalModel(input_dim, n_groups)
    loss_fn = PartialMultivariateLogRankLoss(
        penalty_weight=0.1)  # Novel loss function. Can be used equivalently to any other loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Arbitrary training loop
    for _ in range(n_epochs):
        optimizer.zero_grad()
        scores = model(x)
        loss = loss_fn(scores, true_durations,
                       event_observed)  # Loss function requires soft group assignments along with survival times and event indicators.
        loss.backward()
        optimizer.step()
```

## License

This project is licensed under a GPL-3.0 license - see the [LICENSE](LICENSE) file for details.