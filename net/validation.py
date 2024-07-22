import os
import time
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from net.anchors.gravity_points_prediction import gravity_points_prediction
from net.detections.detections_validation_distance import detections_validation_distance
from net.evaluation.utility.distance_eval_rescale import distance_eval_rescale


def validation(num_epoch: int,
               epochs: int,
               experiment_ID: str,
               net: torch.nn.Module,
               dataloader: DataLoader,
               hook: int,
               gravity_points: torch.Tensor,
               eval: str,
               rescale_factor: float,
               score_threshold: float,
               detections_path: str,
               FP_list_path: str,
               output_gravity_path: str,
               filename_output_gravity: str,
               do_output_gravity: bool,
               device: torch.device
               ):
    """
    Validation function to test the trained model after each epoch on validation dataset

    :param num_epoch: num epoch
    :param epochs: epochs
    :param experiment_ID: experiment ID
    :param net: net
    :param dataloader: dataloader
    :param hook: hook distance
    :param gravity_points: gravity points
    :param eval: eval
    :param rescale_factor: rescale factor
    :param score_threshold: score threshold
    :param detections_path: detections path
    :param FP_list_path: FP list images path
    :param output_gravity_path: output gravity path
    :param filename_output_gravity: filename for output gravity
    :param do_output_gravity: output gravity option
    :param device: device
    """

    # if detections already exists: delete
    if os.path.isfile(detections_path):
        os.remove(detections_path)

    # switch to test mode
    net.eval()

    print("Epoch: {}/{}".format(num_epoch, epochs))

    # do not accumulate gradients (faster)
    with torch.no_grad():

        # for each batch in dataloader
        for num_batch, batch in enumerate(tqdm(dataloader, desc='Validation')):
            # init batch time
            time_batch_start = time.time()

            # get data from dataloader
            filename = batch['filename']
            image = batch['image'].float().to(device)
            mask = batch['image_mask'].float().to(device)
            annotation = batch['annotation'].to(device)

            # output net
            classifications, regressions = net(image=image)

            # compute predictions: gravity points + regression
            predictions = gravity_points_prediction(gravity_points=gravity_points,
                                                    hook=hook,
                                                    regression=regressions)

            # evaluation with distance
            # save detections.csv (with distance metrics)
            detections_validation_distance(filenames=filename,
                                           predictions=predictions,
                                           classifications=classifications,
                                           images=image,
                                           masks=mask,
                                           annotations=annotation,
                                           distance=distance_eval_rescale(eval=eval,
                                                                          rescale=rescale_factor),
                                           score_threshold=score_threshold,
                                           detections_path=detections_path,
                                           FP_list_path=FP_list_path,
                                           filename_output_gravity=filename_output_gravity,
                                           output_gravity_path=os.path.join(output_gravity_path, "{}-output-gravity-epoch={}|{}.png".format(filename_output_gravity, num_epoch, experiment_ID)),
                                           device=device,
                                           do_output_gravity=do_output_gravity)

            # batch time
            time_batch = time.time() - time_batch_start

            # show
            # print("Epoch: {}/{} |".format(num_epoch, epochs),
            #       "Batch: {}/{} |".format(num_batch + 1, len(dataloader)),
            #       "Time: {:.0f} s ".format(int(time_batch) % 60))
