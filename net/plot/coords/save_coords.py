import csv

import numpy as np

from net.initialization.header.coords import coords_header


def save_coords(x: np.ndarray,
                y: np.ndarray,
                coords_type: str,
                path: str):
    """
    Save coords

    :param x: coords x
    :param y: coords y
    :param coords_type: coords type
    :param path: path to save coords
    """

    # save coords
    with open(path, 'w') as file:
        # writer
        writer = csv.writer(file)

        # write header
        header = coords_header(coords_type=coords_type)
        writer.writerow(header)

        # iterate row writer
        for row in range(len(x)):
            writer.writerow([x[row], y[row]])
