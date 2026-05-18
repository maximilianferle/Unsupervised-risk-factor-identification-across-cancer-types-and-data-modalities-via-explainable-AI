from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def remove_top_and_right_spines(ax: plt.Axes) -> plt.Axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def letter_annotation(ax, xoffset, yoffset, letter, size=12):
    ax.text(xoffset, yoffset, f"{letter})",
            transform=ax.transAxes,
            size=size,
            weight='bold')


def assert_partition(partition: str):
    assert partition in ("val_dataset",
                         "test_dataset"), f"Unknown partition: {partition}. Valid partitions are: val_dataset, test_dataset"
