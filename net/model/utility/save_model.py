import torch

from net.utility.msg.msg_save_model_complete import msg_save_best_model_complete
from net.utility.msg.msg_save_model_complete import msg_save_resume_model_complete


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


def save_resume_model(epoch, net, sensitivity_wp, AUFROC_0_10, optimizer, scheduler, path):
    """
    Save resume model

    :param epoch: num epoch
    :param net: net
    :param sensitivity_wp: sensitivity at FPS work point
    :param AUFROC_0_10: AUFROC [0, 10]
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param path: path to save resume model
    """

    # save model
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'sensitivity work point': sensitivity_wp,
        'AUFROC [0, 10]': AUFROC_0_10,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': torch.get_rng_state()
    }, path)

    # msg save resume-model
    msg_save_resume_model_complete(epoch=epoch)
