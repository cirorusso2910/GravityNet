import numpy as np
from matplotlib import pyplot as plt

from net.plot.coords.save_coords import save_coords


def PR_plot(title: str,
            color: str,
            experiment_ID: str,
            precision: np.ndarray,
            recall: np.ndarray,
            PR_path: str,
            PR_coords_path: str):
    """
    Precision-Recall plot

    :param title: plot title
    :param color: plot color
    :param experiment_ID: experiment ID
    :param precision: precision
    :param recall: recall
    :param PR_path: PR path
    :param PR_coords_path: PR coords path
    """

    # Figure: PR
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0, x=0.5)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(precision, recall, color=color)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(PR_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # Save Coords: PR
    save_coords(x=precision,
                y=recall,
                coords_type='PR',
                path=PR_coords_path)
