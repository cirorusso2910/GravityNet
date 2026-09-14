import torch

from net.utility.msg.msg_load_model_complete import msg_load_best_model_complete


def load_best_model(net: torch.nn.Module,
                    metrics_type: str,
                    path: str):
    """
    Load best-model

    :param net: net
    :param metrics_type: metrics type
    :param path: path
    """

    # load model (map to CPU so a GPU-trained checkpoint also loads on CPU-only machines;
    # weights_only=False since the checkpoint stores metrics/optimizer/scheduler state, not just tensors)
    load_model = torch.load(path, map_location='cpu', weights_only=False)

    # load state dict
    net.load_state_dict(load_model['net_state_dict'])

    # msg load best model complete
    msg_load_best_model_complete(metrics_type=metrics_type,
                                 load_model=load_model)
