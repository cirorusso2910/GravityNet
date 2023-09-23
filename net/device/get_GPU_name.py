import sys

import torch

from net.utility.msg.msg_error import msg_error


def get_GPU_name() -> str:
    """
    Get GPU-device name

    :return: GPU-device name
    """

    if torch.cuda.is_available():
        GPU_device_name = torch.cuda.get_device_name(0)

        if 'V100' in GPU_device_name:
            GPU_name = 'V100'

        elif 'A100' in GPU_device_name:
            GPU_name = 'A100'

        else:
            str_err = msg_error(file=__file__,
                                variable=GPU_device_name,
                                type_variable="GPU device name",
                                choices="[V100, A100]")
            sys.exit(str_err)

    else:
        GPU_name = 'cpu'

    return GPU_name
