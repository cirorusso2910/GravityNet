import numpy as np
import torch
import random


def reproducibility(seed: int):
    """
    Set seed for experiment reproducibility

    :param seed: seed
    """

    # for torch
    torch.manual_seed(seed)

    # for numpy
    np.random.seed(seed)

    # for random
    random.seed(seed)

    # for cuda
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
