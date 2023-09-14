import numpy as np

from matplotlib import pyplot as plt

from net.utility.msg.msg_plot_complete import msg_plot_complete


def FROC_linear_plot(title: str,
                     color: str,
                     experiment_ID: str,
                     FPS: np.ndarray,
                     sens: np.ndarray,
                     FROC_upper_limit: int,
                     FROC_path: str):
    """
    FROC Linear plot

    :param title: plot title
    :param color: plot color
    :param experiment_ID: experiment ID
    :param FPS: false positive per scan (FPS)
    :param sens: sensitivity (sens)
    :param FROC_upper_limit: FPS upper limit
    :param FROC_path: FROC path
    """

    # Figure: FROC Linear
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0, x=0.5)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(FPS, sens, color=color)
    plt.xlabel('Average number of false positives per scan')
    plt.xscale('linear')
    plt.xlim(0, FROC_upper_limit)
    plt.xticks(np.arange(0, FROC_upper_limit + 1, 1))
    plt.ylabel('True Positive Rate')
    plt.ylim(0.0, 1.0)
    plt.savefig(FROC_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # plot complete
    msg_plot_complete(plot_type='FROC Linear')
