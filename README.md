# Unsupervised Risk Factor Identification via Explainable Artificial Intelligence

This repository contains the accompanying code for the paper:
**[Unsupervised risk factor identification across cancer types and data modalities via explainable artificial intelligence](https://doi.org/10.1038/s41746-026-02663-w)** by Ferle et al (2026).

## Overview
This repository provides implementations of survival models using the **partial multivariate log-rank loss** for survival-guided clustering. It includes:
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
- **CoMMpass and Lung1 datasets** are **not included** in this repository. Users must obtain their own copies of these datasets from the respective sources:
  - CoMMpass: [https://research.themmrf.org/](https://research.themmrf.org/)
  - Lung1: [https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics)

## Core Innovation: Partial Multivariate Log-Rank Loss
The main innovation of this work is implemented in `loss/partial_multivariate_logrank.py`. This loss module enables model architecture- and data modality-independent clustering of risk groups in a survival-guided manner while handling censored data.

### Example Usage
```python
import torch
from loss.partial_multivariate_logrank import PartialMultivariateLogRankLoss

# Example tensors (batch_size=10, n_groups=3)
scores = torch.rand(10, 3)  # Predicted risk scores
true_durations = torch.rand(10)  # Observed survival durations
event_observed = torch.randint(0, 2, (10,))  # Binary event observations (1 if event occurred, 0 if censored)

# Initialize loss
loss_fn = PartialMultivariateLogRankLoss(penalty_weight=0.1)

# Compute loss
loss = loss_fn(scores, true_durations, event_observed)
print(f"Loss: {loss.item()}")
```

## License
This project is licensed under a GPL-3.0 license - see the [LICENSE](LICENSE) file for details.