import torch

from net.utility.msg.msg_save_model_complete import msg_save_best_model_complete


def save_best_model(epoch, net, metrics, metrics_type, optimizer, scheduler, path):
    """
    Save best model

    :param epoch: num epoch
    :param net: net
    :param metrics: metrics
    :param metrics_type: metrics type
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param path: path to save model
    """

    # save model
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        metrics_type: max(metrics),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': torch.get_rng_state()
    }, path)

    # msg save best-model
    msg_save_best_model_complete(metrics_type=metrics_type)
