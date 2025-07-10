import numpy as np


def AUFROC(FPS: np.ndarray,
           sens: np.ndarray,
           FPS_upper_bound: int) -> float:
    """
    Compute Area Under the FROC Curve (AUFROC) in range [0, FPS-upper-bound]

    :param FPS: false positive per scan (FPS)
    :param sens: sensitivity
    :param FPS_upper_bound: FPS upper bound
    :return: AUFROC value
    """

    x1 = x2 = y1 = y2 = index = 0
    for n in range(len(FPS)):
        if FPS[n] >= FPS_upper_bound:
            x1 = FPS[n - 1]
            x2 = FPS[n]
            y1 = sens[n - 1]
            y2 = sens[n]
            index = n
            break
    if (x2-x1) == 0:
        sens_boundary = np.nan
    else:
        sens_boundary = y1 + (FPS_upper_bound - x1) * (y2 - y1) / (x2 - x1)

    # crop FPS and sens to index of FPS boundary
    FPS_cropped = FPS[0:index]
    sens_cropped = sens[0:index]

    # append FPS boundary and sens boundary (interpolated)
    FPS_cropped = np.append(FPS_cropped, FPS_upper_bound)
    sens_cropped = np.append(sens_cropped, sens_boundary)

    # sort index
    sorted_index = np.argsort(FPS_cropped)
    FPS_cropped_sorted = np.array(FPS_cropped)[sorted_index]
    sens_cropped_sorted = np.array(sens_cropped)[sorted_index]

    # not enough curve points
    if len(FPS_cropped) < 2:
        AUFROC_value = np.nan
    else:
        AUFROC_value = np.trapz(y=sens_cropped_sorted, x=FPS_cropped_sorted) / FPS_upper_bound

    return AUFROC_value
