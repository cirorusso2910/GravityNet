import torch


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
            GPU_name = GPU_device_name

    else:
        GPU_name = 'cpu'

    return GPU_name
