import torch


def allow_tf32(default: bool):
    """
    Starting in PyTorch 1.7, there is a new flag called allow_tf32.

    This flag defaults to True in PyTorch 1.7 to PyTorch 1.11, and False in PyTorch 1.12 and later.

    This flag controls whether PyTorch is allowed to use the TensorFloat32 (TF32) tensor cores,
    available on new NVIDIA GPUs since Ampere,
    internally to compute matmul (matrix multiplies and batched matrix multiplies) and convolutions.

    TF32 tensor cores are designed to achieve better performance on matmul and convolutions on torch.float32 tensors
    by rounding input data to have 10 bits of mantissa,
    and accumulating results with FP32 precision, maintaining FP32 dynamic range

    :param default: default option
    """

    # default in PyTorch 1.12 and later (current version 1.12)
    if default:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True

    # default in PyTorch 1.11 and previous
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("\nALLOW TF 32:"
          "\ntorch.backends.cuda.matmul.allow_tf32: {}"
          "\ntorch.backends.cudnn.allow_tf32: {}".format(torch.backends.cuda.matmul.allow_tf32,
                                                         torch.backends.cudnn.allow_tf32))
