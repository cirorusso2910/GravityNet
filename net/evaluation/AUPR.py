import numpy as np

from scipy import integrate


def AUPR(precision: np.ndarray,
         recall: np.ndarray) -> float:
    """
    Compute Area Under the PR Curve (AUPR)

    :param precision: precision
    :param recall: recall
    :return: AUPR value
    """

    # compute AUPR value
    AUPR_value = integrate.trapz(y=recall, x=precision)

    return AUPR_value
