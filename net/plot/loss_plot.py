from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from net.utility.msg.msg_plot_complete import msg_plot_complete


def loss_plot(figsize: Tuple[int, int],
              title: str,
              experiment_ID: str,
              ticks: List[int],
              epochs_ticks: np.ndarray,
              loss: List[float],
              classification_loss: List[float],
              regression_loss: List[float],
              loss_path: str):
    """
    Loss plot

    :param figsize: figure size
    :param title: plot title
    :param experiment_ID: experiment ID
    :param ticks: ticks
    :param epochs_ticks: epochs ticks
    :param loss: loss
    :param classification_loss: classification loss
    :param regression_loss: regression loss
    :param loss_path: loss path
    """

    # Figure: Loss
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(ticks, loss, marker=".", color='red', label='Loss')
    plt.plot(ticks, classification_loss, marker=".", color='blue', label='Classification loss')
    plt.plot(ticks, regression_loss, marker=".", color='green', label='Regression loss')
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=3)
    plt.xlabel("Epochs")
    plt.xticks(epochs_ticks)
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.savefig(loss_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # plot complete
    msg_plot_complete(plot_type='Loss')
