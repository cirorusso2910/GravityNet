import os
import time
import torch

from torch.utils.data import DataLoader

from net.anchors.gravity_points_prediction import gravity_points_prediction
from net.detections.detections_test_distance import detections_test_distance
from net.evaluation.utility.distance_eval_rescale import distance_eval_rescale


def test(experiment_ID: str,
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
         do_output_gravity: bool,
         do_NMS: bool,
         NMS_box_radius: int,
         device: torch.device
         ):
    """
    Test function on best trained model

    :param experiment_ID: experiment ID
    :param net: net
    :param dataloader: dataloader
    :param hook: hook distance
    :param gravity_points: gravity points
    :param eval: eval
    :param rescale_factor: rescale factor
    :param score_threshold: score threshold
    :param detections_path: detections path
    :param FP_list_path: normals images path
    :param output_gravity_path: output gravity path
    :param do_output_gravity: output gravity option
    :param do_NMS: NMS option
    :param NMS_box_radius: NMS box radius
    :param device: device
    """

    # if detections already exists: delete
    if os.path.isfile(detections_path):
        os.remove(detections_path)

    # switch to test mode
    net.eval()

    # do not accumulate gradients (faster)
    with torch.no_grad():

        # for each batch in dataloader
        for num_batch, batch in enumerate(dataloader):

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
            predictions = gravity_points_prediction(hook=hook,
                                                    gravity_points=gravity_points,
                                                    regression=regressions)

            # evaluation with distance
            # save detections.csv (with distance metrics)
            detections_test_distance(experiment_ID=experiment_ID,
                                     filenames=filename,
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
                                     output_gravity_path=output_gravity_path,
                                     device=device,
                                     do_output_gravity=do_output_gravity,
                                     do_NMS=do_NMS,
                                     NMS_box_radius=NMS_box_radius)

            # batch time
            time_batch = time.time() - time_batch_start

            # show
            print("Batch: {}/{} |".format(num_batch + 1, len(dataloader)),
                  "Time: {:.0f} s ".format(int(time_batch) % 60))
