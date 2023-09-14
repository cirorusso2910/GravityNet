import numpy as np
from matplotlib import pyplot as plt

from net.utility.msg.msg_plot_complete import msg_plot_complete


def score_distribution_plot(title: str,
                            score: np.ndarray,
                            bins: int,
                            experiment_ID: str,
                            score_distribution_path: str):
    """
    Score Distribution plot

    :param title: plot title
    :param score: score
    :param bins: bins
    :param experiment_ID: experiment ID
    :param score_distribution_path: score distribution path
    """

    # Figure: Score Distribution
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(title, fontweight="bold", fontsize=15, y=1.0)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    n, bins, patches = plt.hist(score, bins=bins, log=True)
    plt.xlabel('score')
    plt.ylabel('predictions')
    plt.savefig(score_distribution_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)  # close figure

    # plot complete
    msg_plot_complete(plot_type='Score Distribution')
