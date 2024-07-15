import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time

import numpy as np
import torch
from pandas import read_csv

from torch import nn
from torch.utils.data import DataLoader

from net.anchors.gravity_points_config import gravity_points_config
from net.anchors.utility.check_image_shape import check_image_shape
from net.dataset.classes.dataset_class import dataset_class
from net.dataset.dataset_augmentation import dataset_augmentation
from net.dataset.dataset_num_annotations import dataset_num_annotations
from net.dataset.dataset_num_images import dataset_num_images
from net.dataset.dataset_num_normal_images import dataset_num_normal_images
from net.dataset.dataset_split import dataset_split
from net.dataset.dataset_transforms import dataset_transforms
from net.device.get_GPU_name import get_GPU_name
from net.evaluation.AUC import AUC
from net.evaluation.AUFROC import AUFROC
from net.evaluation.AUPR import AUPR
from net.evaluation.FROC import FROC
from net.evaluation.PR import PR
from net.evaluation.ROC import ROC
from net.evaluation.current_learning_rate import current_learning_rate
from net.evaluation.sensitivity import sensitivity
from net.evaluation.utility.select_FP_list_path import select_FP_list_path
from net.evaluation.utility.select_TotalNumOfImages import select_TotalNumOfImages
from net.initialization.ID.experimentID import experimentID
from net.initialization.dict.metrics import metrics_dict
from net.initialization.dict.plot_title import plot_title_dict
from net.initialization.init import initialization
from net.loss.GravityLoss import GravityLoss
from net.metrics.metrics_test import metrics_test_csv
from net.metrics.metrics_train import metrics_train_csv
from net.metrics.show_metrics.show_metrics_test import show_metrics_test
from net.metrics.show_metrics.show_metrics_train import show_metrics_train
from net.metrics.utility.my_notation import scientific_notation
from net.model.gravitynet.GravityNet import GravityNet
from net.model.utility.load_model import load_best_model
from net.model.utility.save_model import save_best_model
from net.optimizer.get_optimizer import get_optimizer
from net.output.output import output
from net.output.utility.select_output_gravity_filename import select_output_gravity_filename
from net.parameters.parameters import parameters_parsing
from net.parameters.parameters_summary import parameters_summary
from net.plot.AUC_plot import AUC_plot
from net.plot.AUFROC_plot import AUFROC_plot
from net.plot.FROC_linear_plot import FROC_linear_plot
from net.plot.FROC_plot import FROC_plot
from net.plot.PR_plot import PR_plot
from net.plot.ROC_plot import ROC_plot
from net.plot.loss_plot import loss_plot
from net.plot.score_distribution_plot import score_distribution_plot
from net.plot.sensitivity_plot import sensitivity_plot
from net.plot.utility.figure_size import figure_size
from net.reproducibility.reproducibility import reproducibility
from net.scheduler.get_scheduler import get_scheduler
from net.test import test
from net.train import train
from net.utility.execution_mode import execution_mode
from net.utility.msg.msg_load_dataset_complete import msg_load_dataset_complete
from net.utility.read_split import read_split
from net.validation import validation


def main():
    """
        | =============== |
        | GRAVITY NETWORK |
        | =============== |

        A GravityNet with the purpose to detect small lesion

    """

    print("| ========================== |\n"
          "|         GRAVITY NET        |\n"
          "| ========================== |\n")

    # ================== #
    # PARAMETERS-PARSING #
    # ================== #
    # command line parameter parsing
    parser = parameters_parsing()

    # execution mode start
    execution_mode(mode=parser.mode,
                   option='start')

    # ============== #
    # INITIALIZATION #
    # ============== #
    print("\n---------------"
          "\nINITIALIZATION:"
          "\n---------------")

    # experiment ID
    experiment_ID = experimentID(typeID=parser.typeID,
                                 parser=parser)

    # initialization
    path = initialization(network_name="GravityNet",
                          experiment_ID=experiment_ID,
                          parser=parser)

    # read data split
    data_split = read_split(path_split=path['dataset']['split'])

    # plot title
    plot_title = plot_title_dict(parser=parser)

    # ====== #
    # DEVICE #
    # ====== #
    print("\n-------"
          "\nDEVICE:"
          "\n-------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:ID -> run on GPU #ID
    torch.set_num_threads(parser.num_threads)
    print("GPU device name: {}".format(get_GPU_name()))

    # =============== #
    # REPRODUCIBILITY #
    # =============== #
    reproducibility(seed=parser.seed)

    # ============ #
    # LOAD DATASET #
    # ============ #
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

    # ============= #
    # DATASET SPLIT #
    # ============= #
    # subset dataset according to data split
    dataset_train, dataset_val, dataset_test = dataset_split(data_split=data_split,
                                                             dataset=dataset)

    # ================== #
    # DATASET NUM IMAGES #
    # ================== #
    # num images for dataset-train, dataset-val, dataset-test
    num_images = dataset_num_images(statistics_path=path['dataset']['statistics'],
                                    small_lesion=parser.small_lesion.upper())

    # ========================= #
    # DATASET NUM NORMAL IMAGES #
    # ========================= #
    # num normal images for dataset-train, dataset-val, dataset-test
    num_normal_images = dataset_num_normal_images(statistics_path=path['dataset']['statistics'],
                                                  small_lesion=parser.small_lesion.upper())

    # ======================= #
    # DATASET NUM ANNOTATIONS #
    # ======================= #
    # num annotations for dataset-train, dataset-val, dataset-test
    num_annotations = dataset_num_annotations(statistics_path=path['dataset']['statistics'],
                                              small_lesion=parser.small_lesion.upper())

    # ================== #
    # DATASET TRANSFORMS #
    # ================== #
    train_transforms, val_transforms, test_transforms = dataset_transforms(normalization=parser.norm,
                                                                           parser=parser,
                                                                           statistics_path=path['dataset']['statistics'])

    # apply dataset transforms
    dataset_train.dataset.transforms = train_transforms
    dataset_val.dataset.transforms = val_transforms
    dataset_test.dataset.transforms = test_transforms

    # ==================== #
    # DATASET AUGMENTATION #
    # ==================== #
    if parser.do_dataset_augmentation:
        # dataset-train augmented
        dataset_train = dataset_augmentation(normalization=parser.norm,
                                             parser=parser,
                                             dataset_train=dataset_train,
                                             statistics_path=path['dataset']['statistics'])

    # ============ #
    # DATA LOADERS #
    # ============ #
    # dataloader-train
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=parser.batch_size_train,
                                  shuffle=True,
                                  num_workers=parser.num_workers,
                                  pin_memory=True)

    # dataloader-val
    dataloader_val = DataLoader(dataset=dataset_val,
                                batch_size=parser.batch_size_val,
                                shuffle=False,
                                num_workers=parser.num_workers,
                                pin_memory=True)

    # dataloader-test
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=parser.batch_size_test,
                                 shuffle=False,
                                 num_workers=parser.num_workers,
                                 pin_memory=True)

    # ============== #
    # GRAVITY POINTS #
    # ============== #
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

    num_gravity_points = gravity_points.shape[0]  # num gravity points (A)
    num_gravity_points_feature_map = gravity_points_feature_map.shape[0]  # num gravity points in a feature map (reference window)

    # ============= #
    # NETWORK MODEL #
    # ============= #
    # net
    net = GravityNet(backbone=parser.backbone,
                     pretrained=parser.pretrained,
                     num_gravity_points_feature_map=num_gravity_points_feature_map)

    # data parallel
    net = nn.DataParallel(module=net)

    # move net to device
    net.to(device)

    # ======= #
    # SUMMARY #
    # ======= #
    parameters_summary(parser=parser,
                       dataset_name=parser.dataset,
                       num_images=num_images,
                       num_images_normals=num_normal_images,
                       num_annotations=num_annotations,
                       small_lesion=parser.small_lesion,
                       image_shape=image_shape,
                       feature_map_shape=feature_map_shape,
                       num_gravity_points=num_gravity_points,
                       num_gravity_points_feature_map=num_gravity_points_feature_map)

    # =========== #
    # MODE: TRAIN #
    # =========== #
    if parser.mode in ['train', 'train_test']:

        # ========= #
        # OPTIMIZER #
        # ========= #
        optimizer = get_optimizer(net_parameters=net.parameters(),
                                  parser=parser)

        # ========= #
        # SCHEDULER #
        # ========= #
        scheduler = get_scheduler(optimizer=optimizer,
                                  parser=parser)

        # ========= #
        # CRITERION #
        # ========= #
        criterion = GravityLoss(alpha=parser.alpha,
                                gamma=parser.gamma,
                                config=parser.config,
                                hook=parser.hook,
                                hook_gap=parser.gap,
                                num_gravity_points_feature_map=num_gravity_points_feature_map,
                                device=device)

        # ==================== #
        # INIT METRICS (TRAIN) #
        # ==================== #
        metrics = metrics_dict(metrics_type='train')

        # training epochs range
        start_epoch_train = 1  # star train
        stop_epoch_train = start_epoch_train + parser.epochs  # stop train

        # for each epoch
        for epoch in range(start_epoch_train, stop_epoch_train):

            # ======== #
            # TRAINING #
            # ======== #
            print("\n---------"
                  "\nTRAINING:"
                  "\n---------")
            time_train_start = time.time()
            loss, classification_loss, regression_loss = train(num_epoch=epoch,
                                                               epochs=parser.epochs,
                                                               net=net,
                                                               dataloader=dataloader_train,
                                                               gravity_points=gravity_points,
                                                               optimizer=optimizer,
                                                               scheduler=scheduler,
                                                               criterion=criterion,
                                                               lambda_factor=parser.lambda_factor,
                                                               device=device,
                                                               parser=parser)
            time_train = time.time() - time_train_start

            # ========== #
            # VALIDATION #
            # ========== #
            print("\n-----------"
                  "\nVALIDATION:"
                  "\n-----------")
            time_val_start = time.time()
            validation(num_epoch=epoch,
                       epochs=parser.epochs,
                       experiment_ID=experiment_ID,
                       net=net,
                       dataloader=dataloader_val,
                       hook=parser.hook,
                       gravity_points=gravity_points,
                       eval=parser.eval,
                       rescale_factor=parser.rescale,
                       score_threshold=parser.score_threshold,
                       detections_path=path['detections']['validation'],
                       FP_list_path=select_FP_list_path(FP_images=parser.FP_images,
                                                        path=path['dataset']),
                       filename_output_gravity=select_output_gravity_filename(dataset=dataset_val,
                                                                              idx=parser.idx),
                       output_gravity_path=path['output']['gravity']['validation'],
                       do_output_gravity=parser.do_output_gravity,
                       device=device)
            time_val = time.time() - time_val_start

            # ==================== #
            # METRICS (VALIDATION) #
            # ==================== #
            time_metrics_val_start = time.time()

            # read detections validation for evaluation (numpy array)
            detections_val = read_csv(filepath_or_buffer=path['detections']['validation'], usecols=["LABEL", "SCORE"]).dropna(subset='LABEL').values

            # compute AUC
            AUC_val = AUC(detections=detections_val)

            # compute FROC
            FPS_val, sens_val = FROC(detections=detections_val,
                                     TotalNumOfImages=select_TotalNumOfImages(FP_images=parser.FP_images,
                                                                              num_images=num_images['validation'],
                                                                              num_images_normals=num_normal_images['validation']),
                                     TotalNumOfAnnotations=num_annotations['validation'])

            # compute sensitivity 10 FPS
            sens_10_FPS_val, sens_max_val = sensitivity(FPS=FPS_val,
                                                        sens=sens_val,
                                                        work_point=10)

            # compute ROC
            FPR_val, TPR_val = ROC(detections=detections_val)

            # compute AUFROC
            AUFROC_0_1_val = AUFROC(FPS=FPS_val, sens=sens_val, FPS_upper_bound=1)
            AUFROC_0_10_val = AUFROC(FPS=FPS_val, sens=sens_val, FPS_upper_bound=10)
            AUFROC_0_50_val = AUFROC(FPS=FPS_val, sens=sens_val, FPS_upper_bound=50)
            AUFROC_0_100_val = AUFROC(FPS=FPS_val, sens=sens_val, FPS_upper_bound=100)

            # compute PR
            precision_val, recall_val = PR(detections=detections_val)

            # compute AUPR
            AUPR_val = AUPR(precision=precision_val, recall=recall_val)

            # =============== #
            # PLOT VALIDATION #
            # =============== #
            print("\n----------------"
                  "\nPLOT VALIDATION:"
                  "\n----------------")
            # FROC plot for each epoch
            FROC_path = os.path.join(path['plots_validation']['FROC'], "FROC-validation-ep={}|".format(epoch) + experiment_ID + ".png")
            FROC_coords_path = os.path.join(path['plots_validation']['coords_FROC'], "FROC-validation-ep={}-coords|".format(epoch) + experiment_ID + ".csv")
            FROC_plot(title="FROC (VALIDATION) | EPOCH={}".format(epoch),
                      color='green',
                      experiment_ID=experiment_ID,
                      FPS=FPS_val,
                      sens=sens_val,
                      FROC_path=FROC_path,
                      FROC_coords_path=FROC_coords_path)

            # ROC plot for each epoch
            ROC_path = os.path.join(path['plots_validation']['ROC'], "ROC-validation-ep={}|".format(epoch) + experiment_ID + ".png")
            ROC_coords_path = os.path.join(path['plots_validation']['coords_ROC'], "ROC-validation-ep={}-coords|".format(epoch) + experiment_ID + ".csv")
            ROC_plot(title="ROC (VALIDATION) | EPOCH={}".format(epoch),
                     color='green',
                     experiment_ID=experiment_ID,
                     FPR=FPR_val,
                     TPR=TPR_val,
                     ROC_path=ROC_path,
                     ROC_coords_path=ROC_coords_path)

            # PR plot for each epoch
            PR_path = os.path.join(path['plots_validation']['PR'], "PR-validation-ep={}|".format(epoch) + experiment_ID + ".png")
            PR_coords_path = os.path.join(path['plots_validation']['coords_PR'], "PR-validation-ep={}-coords|".format(epoch) + experiment_ID + ".csv")
            PR_plot(title="PR (VALIDATION) | EPOCH={}".format(epoch),
                    color='green',
                    experiment_ID=experiment_ID,
                    precision=precision_val,
                    recall=recall_val,
                    PR_path=PR_path,
                    PR_coords_path=PR_coords_path)

            # get current learning rate
            last_learning_rate = current_learning_rate(scheduler=scheduler,
                                                       optimizer=optimizer,
                                                       parser=parser)

            time_metrics_val = time.time() - time_metrics_val_start

            # update performance
            metrics['ticks'].append(epoch)
            metrics['loss']['loss'].append(loss)
            metrics['loss']['classification'].append(classification_loss)
            metrics['loss']['regression'].append(regression_loss)
            metrics['learning_rate'].append(scientific_notation(number=last_learning_rate))
            metrics['AUC'].append(AUC_val)
            metrics['sensitivity']['10 FPS'].append(sens_10_FPS_val)
            metrics['sensitivity']['max'].append(sens_max_val)
            metrics['AUFROC']['[0, 1]'].append(AUFROC_0_1_val)
            metrics['AUFROC']['[0, 10]'].append(AUFROC_0_10_val)
            metrics['AUFROC']['[0, 50]'].append(AUFROC_0_50_val)
            metrics['AUFROC']['[0, 100]'].append(AUFROC_0_100_val)
            metrics['AUPR'].append(AUPR_val),
            metrics['time']['train'].append(time_train)
            metrics['time']['validation'].append(time_val)
            metrics['time']['metrics'].append(time_metrics_val)

            # metrics-train.csv
            metrics_train_csv(metrics_path=path['metrics']['train'],
                              metrics=metrics)

            # show metrics train
            show_metrics_train(metrics=metrics)

            # =============== #
            # SAVE BEST MODEL #
            # =============== #
            print("\n----------------"
                  "\nSAVE BEST MODEL:"
                  "\n----------------")
            # save best-model with sensitivity 10 FPS
            if (epoch - 1) == np.argmax(metrics['sensitivity']['10 FPS']):
                save_best_model(epoch=epoch,
                                net=net,
                                metrics=metrics['sensitivity']['10 FPS'],
                                metrics_type='sensitivity 10 FPS',
                                optimizer=optimizer,
                                scheduler=scheduler,
                                path=path['model']['best']['sensitivity']['10 FPS'])

            # save best-model with AUFROC [0, 10] metrics
            if (epoch - 1) == np.argmax(metrics['AUFROC']['[0, 10]']):
                save_best_model(epoch=epoch,
                                net=net,
                                metrics=metrics['AUFROC']['[0, 10]'],
                                metrics_type='AUFROC [0, 10]',
                                optimizer=optimizer,
                                scheduler=scheduler,
                                path=path['model']['best']['AUFROC']['[0, 10]'])

            # save best-model with AUPR metrics
            if (epoch - 1) == np.argmax(metrics['AUPR']):
                save_best_model(epoch=epoch,
                                net=net,
                                metrics=metrics['AUPR'],
                                metrics_type='AUPR',
                                optimizer=optimizer,
                                scheduler=scheduler,
                                path=path['model']['best']['AUPR'])

            # ========== #
            # PLOT TRAIN #
            # ========== #
            print("\n-----------"
                  "\nPLOT TRAIN:"
                  "\n-----------")
            # figure size
            figsize_x, figsize_y = figure_size(epochs=parser.epochs)

            # epochs ticks
            epochs_ticks = np.arange(1, parser.epochs + 1, step=1)

            # Loss plot
            loss_plot(figsize=(figsize_x, figsize_y),
                      title=plot_title['plots_train']['loss'],
                      experiment_ID=experiment_ID,
                      ticks=metrics['ticks'],
                      epochs_ticks=epochs_ticks,
                      loss=metrics['loss']['loss'],
                      classification_loss=metrics['loss']['classification'],
                      regression_loss=metrics['loss']['regression'],
                      loss_path=path['plots_train']['loss'])

            # Sensitivity plot
            sensitivity_plot(figsize=(figsize_x, figsize_y),
                             title=plot_title['plots_validation']['sensitivity'],
                             experiment_ID=experiment_ID,
                             ticks=metrics['ticks'],
                             epochs_ticks=epochs_ticks,
                             sensitivity_10_FPS=metrics['sensitivity']['10 FPS'],
                             sensitivity_max=metrics['sensitivity']['max'],
                             sensitivity_path=path['plots_validation']['sensitivity'])

            # AUC plot
            AUC_plot(figsize=(figsize_x, figsize_y),
                     title=plot_title['plots_validation']['AUC'],
                     experiment_ID=experiment_ID,
                     ticks=metrics['ticks'],
                     epochs_ticks=epochs_ticks,
                     AUC=metrics['AUC'],
                     AUC_path=path['plots_validation']['AUC'])

            # AUFROC plot
            AUFROC_plot(figsize=(figsize_x, figsize_y),
                        title=plot_title['plots_validation']['AUFROC'],
                        experiment_ID=experiment_ID,
                        ticks=metrics['ticks'],
                        epochs_ticks=epochs_ticks,
                        AUFROC_0_1=metrics['AUFROC']['[0, 1]'],
                        AUFROC_0_10=metrics['AUFROC']['[0, 10]'],
                        AUFROC_0_50=metrics['AUFROC']['[0, 50]'],
                        AUFROC_0_100=metrics['AUFROC']['[0, 100]'],
                        AUFROC_val_path=path['plots_validation']['AUFROC'])

    # ========== #
    # MODE: TEST #
    # ========== #
    if parser.mode in ['test', 'train_test']:

        # =================== #
        # INIT METRICS (TEST) #
        # =================== #
        metrics = metrics_dict(metrics_type='test')

        # =============== #
        # LOAD BEST MODEL #
        # =============== #
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

        # ==== #
        # TEST #
        # ==== #
        print("\n-----"
              "\nTEST:"
              "\n-----")
        time_test_start = time.time()
        test(experiment_ID=experiment_ID,
             net=net,
             dataloader=dataloader_test,
             hook=parser.hook,
             gravity_points=gravity_points,
             eval=parser.eval,
             rescale_factor=parser.rescale,
             score_threshold=parser.score_threshold,
             detections_path=path['detections']['test'],
             FP_list_path=select_FP_list_path(FP_images=parser.FP_images,
                                              path=path['dataset']),
             output_gravity_path=path['output']['gravity']['test'],
             do_output_gravity=parser.do_output_gravity,
             do_NMS=parser.do_NMS,
             NMS_box_radius=parser.NMS_box_radius,
             device=device)
        time_test = time.time() - time_test_start

        # read detections test for evaluation (numpy array)
        detections_test = read_csv(filepath_or_buffer=path['detections']['test'], usecols=["LABEL", "SCORE"]).dropna(subset='LABEL').values
        detections_score = detections_test[:, 1]  # detections score

        # ============== #
        # METRICS (TEST) #
        # ============== #
        time_metrics_test_start = time.time()

        # compute AUC
        AUC_test = AUC(detections=detections_test)

        # compute FROC
        FPS_test, sens_test = FROC(detections=detections_test,
                                   TotalNumOfImages=select_TotalNumOfImages(FP_images=parser.FP_images,
                                                                            num_images=num_images['test'],
                                                                            num_images_normals=num_normal_images['test']),
                                   TotalNumOfAnnotations=num_annotations['test'])

        # compute sensitivity 10 FPS
        sens_10_FPS_test, sens_max_test = sensitivity(FPS=FPS_test,
                                                      sens=sens_test,
                                                      work_point=10)

        # compute AUFROC
        AUFROC_0_1_test = AUFROC(FPS=FPS_test, sens=sens_test, FPS_upper_bound=1)
        AUFROC_0_10_test = AUFROC(FPS=FPS_test, sens=sens_test, FPS_upper_bound=10)
        AUFROC_0_50_test = AUFROC(FPS=FPS_test, sens=sens_test, FPS_upper_bound=50)
        AUFROC_0_100_test = AUFROC(FPS=FPS_test, sens=sens_test, FPS_upper_bound=100)

        # compute ROC
        FPR_test, TPR_test = ROC(detections=detections_test)

        # compute PR
        precision_test, recall_test = PR(detections=detections_test)

        # compute AUPR
        AUPR_test = AUPR(precision=precision_test, recall=recall_test)

        time_metrics_test = time.time() - time_metrics_test_start

        # update performance
        metrics['AUC'].append(AUC_test)
        metrics['sensitivity']['10 FPS'].append(sens_10_FPS_test)
        metrics['sensitivity']['max'].append(sens_max_test)
        metrics['AUFROC']['[0, 1]'].append(AUFROC_0_1_test)
        metrics['AUFROC']['[0, 10]'].append(AUFROC_0_10_test)
        metrics['AUFROC']['[0, 50]'].append(AUFROC_0_50_test)
        metrics['AUFROC']['[0, 100]'].append(AUFROC_0_100_test)
        metrics['AUPR'].append(AUPR_test)
        metrics['time']['test'].append(time_test)
        metrics['time']['metrics'].append(time_metrics_test)

        # metrics-test.csv
        metrics_test_csv(metrics_path=path['metrics']['test'],
                         metrics=metrics)

        # show metrics test
        show_metrics_test(metrics=metrics)

        # ====== #
        # OUTPUT #
        # ====== #
        print("\n-------"
              "\nOUTPUT:"
              "\n-------")
        output(type_draw=parser.type_draw,
               box_draw_radius=parser.box_draw_radius,
               dataset=dataset_test,
               num_images=parser.num_images,
               detections_path=path['detections']['test'],
               output_path=path['output']['test'],
               suffix="-output|{}".format(experiment_ID))

        # ========= #
        # PLOT TEST #
        # ========= #
        print("\n----------"
              "\nPLOT TEST:"
              "\n----------")
        # FROC plot
        FROC_plot(title=plot_title['plots_test']['FROC'],
                  color='green',
                  experiment_ID=experiment_ID,
                  FPS=FPS_test,
                  sens=sens_test,
                  FROC_path=path['plots_test']['FROC'],
                  FROC_coords_path=path['plots_test']['coords']['FROC'])

        # FROC linear plot
        FROC_linear_plot(title=plot_title['plots_test']['FROC'],
                         color='green',
                         experiment_ID=experiment_ID,
                         FPS=FPS_test,
                         sens=sens_test,
                         FROC_upper_limit=10,
                         FROC_path=path['plots_test']['FROC_linear'])

        # ROC plot
        ROC_plot(title=plot_title['plots_test']['ROC'],
                 color='green',
                 experiment_ID=experiment_ID,
                 FPR=FPR_test,
                 TPR=TPR_test,
                 ROC_path=path['plots_test']['ROC'],
                 ROC_coords_path=path['plots_test']['coords']['ROC'])

        # Score Distribution
        score_distribution_plot(title=plot_title['plots_test']['score_distribution'],
                                score=detections_score,
                                bins=10000,
                                experiment_ID=experiment_ID,
                                score_distribution_path=path['plots_test']['score_distribution'])


if __name__ == "__main__":
    main()
