"""
Utilities to interpret, plot figures
"""

import pandas as pd
from matplotlib import pyplot as plt


def plot_loss_curve(classifiers, labels=None):
    """
    Plot train and validation loss curves for one or more classifiers.

    Parameters
    ----------
    classifiers : list
        List of fitted classifiers with a skorch/braindecode-style `history`.
    labels : list[str], optional
        Names to use for each classifier in the legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    if not isinstance(classifiers, (list, tuple)):
        classifiers = [classifiers]

    if labels is None:
        labels = [f"Classifier {i + 1}" for i in range(len(classifiers))]

    if len(labels) != len(classifiers):
        raise ValueError("`labels` must have the same length as `classifiers`.")

    fig, ax = plt.subplots(figsize=(8, 5))

    results_columns = ["train_loss", "valid_loss"]

    for clf, label in zip(classifiers, labels):
        df = pd.DataFrame(
            clf.history[:, results_columns],
            columns=results_columns,
            index=clf.history[:, "epoch"],
        )

        ax.plot(
            df.index,
            df["train_loss"],
            linestyle="-",
            label=f"{label} train loss",
        )

        ax.plot(
            df.index,
            df["valid_loss"],
            linestyle="--",
            label=f"{label} valid loss",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig
