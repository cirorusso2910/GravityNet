import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import numpy as np
from torch import nn


from net.anchors.gravity_points_config import gravity_points_config
from net.anchors.utility.check_image_shape import check_image_shape
from net.dataset.classes.dataset_class import dataset_class
from net.dataset.dataset_split import dataset_split
from net.dataset.dataset_transforms import dataset_transforms
from net.device.get_GPU_name import get_GPU_name
from net.explainability.MyGradCAM import MyGradCAM
from net.explainability.utility.save_heatmap import save_heatmap
from net.explainability.utility.save_image_overlay import save_image_overlay
from net.initialization.ID.experimentID import experimentID
from net.initialization.init import initialization
from net.initialization.utility.create_folder import create_folder
from net.loss.GravityLoss import GravityLoss
from net.model.gravitynet.GravityNet import GravityNet
from net.model.utility.load_model import load_best_model
from net.parameters.parameters import parameters_parsing
from net.reproducibility.reproducibility import reproducibility
from net.utility.execution_mode import execution_mode
from net.utility.msg.msg_error import msg_error
from net.utility.msg.msg_load_dataset_complete import msg_load_dataset_complete
from net.utility.read_split import read_split


def main():
    """
        | ------------------------- |
        | EXPLAINABILITY - Grad-CAM |
        | ------------------------- |

        Apply Explainable AI Grad-CAM method

    """

    print("| ------------------------- |\n"
          "| EXPLAINABILITY - Grad-CAM |\n"
          "| ------------------------- |\n")

    # ------------------ #
    # PARAMETERS-PARSING #
    # ------------------ #
    # command line parameter parsing
    parser = parameters_parsing()

    # execution mode start
    execution_mode(mode=parser.mode,
                   option='start')

    # -------------- #
    # INITIALIZATION #
    # -------------- #
    print("\n---------------"
          "\nINITIALIZATION:"
          "\n---------------")

    # experiment ID
    experiment_ID = experimentID(typeID=parser.typeID,
                                 sep=parser.sep,
                                 parser=parser)

    # initialization
    path = initialization(network_name="GravityNet",
                          experiment_ID=experiment_ID,
                          parser=parser)

    # read data split
    data_split = read_split(path_split=path['dataset']['split'])

    # ------ #
    # DEVICE #
    # ------ #
    print("\n-------"
          "\nDEVICE:"
          "\n-------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:ID -> run on GPU #ID
    torch.set_num_threads(parser.num_threads)
    print("GPU device name: {}".format(get_GPU_name()))

    # --------------- #
    # REPRODUCIBILITY #
    # --------------- #
    reproducibility(seed=parser.seed)

    # ------------ #
    # LOAD DATASET #
    # ------------ #
    print("\n-------------"
          "\nLOAD DATASET:"
          "\n-------------")
    dataset = dataset_class(images_dir=path['dataset']['images']['all'],
                            images_extension=parser.images_extension,
                            images_masks_dir=path['dataset']['images']['masks'],
                            images_masks_extension=parser.images_masks_extension,
                            annotations_dir=path['dataset']['annotations']['all'],
                            annotations_extension=parser.annotations_extension,
                            filename_list=data_split['filename'],
                            transforms=None)

    msg_load_dataset_complete(dataset_name=parser.dataset)

    # ------------- #
    # DATASET SPLIT #
    # ------------- #
    # subset dataset according to data split
    dataset_train, dataset_val, dataset_test = dataset_split(data_split=data_split,
                                                             dataset=dataset)

    # ------------------ #
    # DATASET TRANSFORMS #
    # ------------------ #
    train_transforms, val_transforms, test_transforms = dataset_transforms(normalization=parser.norm,
                                                                           parser=parser,
                                                                           statistics_path=path['dataset']['statistics'])

    # apply dataset transforms
    dataset_train.dataset.transforms = train_transforms
    dataset_val.dataset.transforms = val_transforms
    dataset_test.dataset.transforms = test_transforms

    # -------------- #
    # GRAVITY POINTS #
    # -------------- #
    print("\n---------------"
          "\nGRAVITY POINTS:"
          "\n---------------")
    # image shape (C x H x W)
    image_channels, image_height, image_width = dataset_train[0]['image'].shape  # get image shape
    image_shape = np.array((int(image_height), int(image_width)))  # converts to numpy.array

    # check image shape dimension
    check_image_shape(image_shape=image_shape)

    # generate gravity points
    gravity_points, gravity_points_feature_map, feature_map_shape = gravity_points_config(config=parser.config,
                                                                                          image_shape=image_shape,
                                                                                          device=device)

    num_gravity_points_feature_map = gravity_points_feature_map.shape[0]  # num gravity points in a feature map (reference window)

    # ------------- #
    # NETWORK MODEL #
    # ------------- #
    # net
    net = GravityNet(backbone=parser.backbone,
                     pretrained=parser.pretrained,
                     num_gravity_points_feature_map=num_gravity_points_feature_map)

    # data parallel
    net = nn.DataParallel(module=net)

    # move net to device
    net.to(device)

    # --------------- #
    # LOAD BEST MODEL #
    # --------------- #
    print("\n----------------"
          "\nLOAD BEST MODEL:"
          "\n----------------")

    # load best model sensitivity 10 FPS
    if parser.load_best_sensitivity_10_FPS_model:
        load_best_model(net=net,
                        metrics_type='sensitivity 10 FPS',
                        path=path['model']['best']['sensitivity']['10 FPS'])

    # load best model AUFROC [0, 10]
    if parser.load_best_AUFROC_0_10_model:
        load_best_model(net=net,
                        metrics_type='AUFROC [0, 10]',
                        path=path['model']['best']['AUFROC']['[0, 10]'])

    # load best model AUPR
    if parser.load_best_AUPR_model:
        load_best_model(net=net,
                        metrics_type='AUPR',
                        path=path['model']['best']['AUPR'])

    # --------- #
    # CRITERION #
    # --------- #
    criterion = GravityLoss(alpha=parser.alpha,
                            gamma=parser.gamma,
                            config=parser.config,
                            hook=parser.hook,
                            hook_gap=parser.gap,
                            num_gravity_points_feature_map=num_gravity_points_feature_map,
                            device=device)

    # ----------- #
    # READ SAMPLE #
    # ----------- #
    for i in range(parser.num_images):

        # sample
        sample = dataset_test[i]

        # filename
        filename = sample['filename']

        # read image
        image = sample['image'].to(device)
        image = image.unsqueeze(0)  # add batch (B) dimension
        image = image.to(device)

        # read annotation
        annotation = sample['annotation'].to(device)
        annotation = annotation.unsqueeze(0)  # add batch (B) dimension
        annotation = annotation.to(device)

        # switch to eval mode
        net.eval()

        # -------------- #
        # EXPLAINABILITY #
        # -------------- #
        print("\n---------------"
              "\nEXPLAINABILITY:"
              "\n---------------")

        # define target layer for explainability in ResNet model
        if 'ResNet' in parser.backbone:
            target_layer = net.module.backboneModel.layer4

        # define target layer for explainability in Swin model
        elif 'Swin' in parser.backbone:
            target_layer = net.module.backboneModel.norm

        else:
            str_err = msg_error(file=__file__,
                                variable=parser.backbone,
                                type_variable="backbone (target layer)",
                                choices="[ResNet, Swin]")
            sys.exit(str_err)

        # MyGradCAM
        my_gradcam = MyGradCAM(backbone=parser.backbone)
        target_layer.register_forward_hook(my_gradcam.forward_hook)
        target_layer.register_full_backward_hook(my_gradcam.backward_hook)

        # forward pass
        classifications, regressions = net(image)

        # calculate loss
        classification_loss, regression_loss = criterion(images=image,
                                                         classifications=classifications,
                                                         regressions=regressions,
                                                         gravity_points=gravity_points,
                                                         annotations=annotation)

        # compute the final loss
        loss = classification_loss + parser.lambda_factor * regression_loss

        # reset parameters gradients of the model (net)
        net.zero_grad()

        # loss gradient backpropagation
        loss.backward()

        # ------------------ #
        # GradCam GravityNet #
        # ------------------ #
        print("\n-------------------"
              "\nGradCAM GravityNet:"
              "\n-------------------")
        # heatmap
        heatmap = my_gradcam.heatmap()

        # ------------ #
        # SAVE HEATMAP #
        # ------------ #
        # init GradCam result folder
        GradCam_result_folder = os.path.join(str(path['explainability']), "GradCAM")
        create_folder(path=GradCam_result_folder)

        # init filename-image result folder
        filename_result_folder = os.path.join(GradCam_result_folder, filename)
        create_folder(path=filename_result_folder)

        heatmap_path = os.path.join(filename_result_folder, "heatmap.png")
        image_overlay_path = os.path.join(filename_result_folder, "image_overlay.png")

        # show heatmap
        save_heatmap(heatmap=heatmap,
                     output_path=heatmap_path,
                     scale_factor=5)

        # show image overlay
        save_image_overlay(image=image.squeeze(),
                           heatmap=heatmap,
                           size=(image_shape[1], image_shape[0]),
                           output_path=image_overlay_path)

        print("GradCAM GravityNet - Complete")

    # execution mode complete
    execution_mode(mode=parser.mode,
                   option='complete')


if __name__ == "__main__":
    main()
