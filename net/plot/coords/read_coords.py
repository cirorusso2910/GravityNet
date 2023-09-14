import numpy as np

from pandas import read_csv
from typing import Tuple

from net.initialization.header.coords import coords_header


def read_coords(coords_path: str,
                coords_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read coords from path according to type

    :param coords_path: coords path
    :param coords_type: coords type
    :return: coords x,
             coords y
    """

    # read coords
    header = coords_header(coords_type=coords_type)
    coords = read_csv(filepath_or_buffer=coords_path, usecols=header, float_precision='round_trip').values

    coord_x = coords[:, 0]  # np.ndarray
    coord_y = coords[:, 1]  # np.ndarray

    return coord_x, coord_y
