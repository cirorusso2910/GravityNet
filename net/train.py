import argparse
import time

from typing import Union, Tuple

import numpy as np
import torch

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.loss.GravityLoss import GravityLoss
from net.metrics.utility.timer import timer


def train(num_epoch: int,
          epochs: int,
          net: torch.nn.Module,
          dataloader: DataLoader,
          gravity_points: torch.Tensor,
          optimizer: Union[Adam, SGD],
          scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
          criterion: Union[GravityLoss],
          lambda_factor: int,
          device: torch.device,
          parser: argparse.Namespace) -> Tuple[float, float, float]:
    """
    Training function

    :param dataset: dataset name
    :param num_epoch: num epoch
    :param epochs: epochs
    :param net: net
    :param dataloader: dataloader
    :param gravity_points: gravity points
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param criterion: criterion (loss)
    :param lambda_factor: lambda factor
    :param device: device
    :param parser: parser of parameters-parsing
    :return: average epoch loss,
             average classification loss,
             average regression loss
    """

    # switch to train mode
    net.train()

    # reset performance measures
    epoch_loss_hist = []
    classification_loss_hist = []
    regression_loss_hist = []

    # for each batch in dataloader
    for num_batch, batch in enumerate(tqdm(dataloader, desc='Training')):

        # init batch time
        time_batch_start = time.time()

        # get data from dataloader
        image = batch['image'].to(device)
        annotation = batch['annotation'].to(device)

        # zero (init) the parameter gradients
        optimizer.zero_grad()

        # forward pass
        classifications, regressions = net(image)

        # calculate loss
        classification_loss, regression_loss = criterion(images=image,
                                                         classifications=classifications,
                                                         regressions=regressions,
                                                         gravity_points=gravity_points,
                                                         annotations=annotation)

        # compute the final loss
        loss = classification_loss + lambda_factor * regression_loss

        # append subnet loss (classification and regression)
        classification_loss_hist.append(float(classification_loss))
        regression_loss_hist.append(float(regression_loss))

        # append epoch loss
        epoch_loss_hist.append(float(loss))

        # loss gradient backpropagation
        loss.backward()

        # clip gradient
        if parser.clip_gradient:
            clip_grad_norm_(parameters=net.parameters(),
                            max_norm=parser.max_norm)

        # net parameters update
        optimizer.step()

        # batch time
        time_batch = time.time() - time_batch_start

        # batch time conversion
        batch_time = timer(time_elapsed=time_batch)

        # print("Epoch: {}/{} |".format(num_epoch, epochs),
        #       "Batch: {}/{} |".format(num_batch + 1, len(dataloader)),
        #       "Classification Loss: {:1.5f} |".format(float(classification_loss)),
        #       "Regression Loss: {:1.5f} |".format(float(regression_loss)),
        #       "Loss: {:1.5f} |".format(float(loss)),
        #       "Time: {:.0f} s ".format(batch_time['seconds']))

        del classification_loss
        del regression_loss
        del loss
        del time_batch

    # step learning rate scheduler
    if parser.scheduler == 'ReduceLROnPlateau':
        scheduler.step(np.mean(epoch_loss_hist))
    elif parser.scheduler == 'StepLR':
        scheduler.step()

    # return avg epoch loss, avg classification loss, avg regression loss (for each epoch)
    return sum(epoch_loss_hist) / len(dataloader), sum(classification_loss_hist) / len(dataloader), sum(regression_loss_hist) / len(dataloader)
