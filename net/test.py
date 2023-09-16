import os
import sys
import time
import torch

from torch.utils.data import DataLoader

from net.anchors.gravity_points_prediction import gravity_points_prediction
from net.detections.detections_test_distance import detections_test_distance
from net.detections.detections_test_radius import detections_test_radius
from net.evaluation.utility.distance_eval_rescale import distance_eval_rescale
from net.evaluation.utility.radius_eval import radius_eval
from net.utility.msg.msg_error import msg_error


def test(experiment_ID: str,
         net: torch.nn.Module,
         dataloader: DataLoader,
         hook: int,
         gravity_points: torch.Tensor,
         eval: str,
         rescale_factor: float,
         detections_path: str,
         FP_list_path: str,
         output_gravity_path: str,
         do_output_gravity: bool,
         device: torch.device,
         debug: bool):
    """
    Test function on best trained model

    :param experiment_ID: experiment ID
    :param net: net
    :param dataloader: dataloader
    :param hook: hook distance
    :param gravity_points: gravity points
    :param eval: eval
    :param rescale_factor: rescale factor
    :param detections_path: detections path
    :param FP_list_path: normals images path
    :param output_gravity_path: output gravity path
    :param do_output_gravity: output gravity option
    :param device: device
    :param debug: debug option
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
            filename, image, mask, annotation = batch['filename'], batch['image'].float().to(device), batch['image_mask'].float().to(device), batch['annotation'].to(device)

            # output net
            classifications, regressions = net(image=image)

            # compute predictions: gravity points + regression
            predictions = gravity_points_prediction(hook=hook,
                                                    gravity_points=gravity_points,
                                                    regression=regressions)

            # evaluation with distance
            if 'distance' in eval:
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
                                         detections_path=detections_path,
                                         FP_list_path=FP_list_path,
                                         output_gravity_path=output_gravity_path,
                                         device=device,
                                         do_output_gravity=do_output_gravity,
                                         debug=debug)

            # evaluation with radius
            elif 'radius' in eval:
                # save detections.csv (with radius metrics)
                detections_test_radius(experiment_ID=experiment_ID,
                                       filenames=filename,
                                       predictions=predictions,
                                       classifications=classifications,
                                       images=image,
                                       masks=mask,
                                       annotations=annotation,
                                       factor=radius_eval(eval=eval),
                                       detections_path=detections_path,
                                       FP_list_path=FP_list_path,
                                       output_gravity_path=output_gravity_path,
                                       device=device,
                                       do_output_gravity=do_output_gravity,
                                       debug=debug)

            else:
                str_err = msg_error(file=__file__,
                                    variable=eval,
                                    type_variable="evaluation",
                                    choices="[distance, radius]")
                sys.exit(str_err)

            # batch time
            time_batch = time.time() - time_batch_start

            # show
            print("Batch: {}/{} |".format(num_batch + 1, len(dataloader)),
                  "Time: {:.0f} s ".format(int(time_batch) % 60))
