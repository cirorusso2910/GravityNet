import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import time

import numpy as np
import torch
from pandas import read_csv

from torch import nn
from torch.utils.data import DataLoader

from net.anchors.gravity_points_config import gravity_points_config
from net.dataset.dataset import DatasetName
from net.dataset.dataset_augmentation import dataset_augmentation
from net.dataset.dataset_num_annotations import dataset_num_annotations
from net.dataset.dataset_num_images import dataset_num_images
from net.dataset.dataset_num_normal_images import dataset_num_normal_images
from net.dataset.dataset_split import dataset_split
from net.dataset.dataset_transforms import dataset_transforms
from net.debug.debug_execution import debug_execution
from net.device.get_GPU_name import get_GPU_name
from net.evaluation.AUC import AUC
from net.evaluation.AUFROC import AUFROC
from net.evaluation.FROC import FROC
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
from net.metrics.metrics_test_NMS import metrics_test_NMS_csv
from net.metrics.metrics_train import metrics_train_csv
from net.metrics.show_metrics.show_metrics_test import show_metrics_test
from net.metrics.show_metrics.show_metrics_test_NMS import show_metrics_test_NMS
from net.metrics.show_metrics.show_metrics_train import show_metrics_train
from net.metrics.utility.my_notation import scientific_notation
from net.model.gravitynet.GravityNet import GravityNet
from net.model.utility.load_model import load_resume_model, check_load_model, load_best_model
from net.model.utility.save_model import save_resume_model, save_best_model
from net.optimizer.get_optimizer import get_optimizer
from net.output.output import output
from net.output.utility.select_output_gravity_filename import select_output_gravity_filename
from net.parameters.parameters import parameters_parsing
from net.parameters.parameters_summary import parameters_summary
from net.plot.AUC_plot import AUC_plot
from net.plot.AUFROC_plot import AUFROC_plot
from net.plot.FROC_linear_plot import FROC_linear_plot
from net.plot.FROC_plot import FROC_plot
from net.plot.ROC_plot import ROC_plot
from net.plot.loss_plot import loss_plot
from net.plot.score_distribution_plot import score_distribution_plot
from net.plot.sensitivity_plot import sensitivity_plot
from net.plot.utility.figure_size import figure_size
from net.reproducibility.reproducibility import reproducibility
from net.resume.metrics_resume import metrics_resume
from net.resume.metrics_train_resume import metrics_train_resume_csv
from net.resume.resume_output import resume_output_validation
from net.resume.resume_plot import resume_ROC_plot, resume_FROC_plot
from net.scheduler.get_scheduler import get_scheduler
from net.test import test
from net.test_NMS import test_NMS
from net.train import train
from net.utility.execution_mode import execution_mode
from net.utility.msg.msg_error import msg_error
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
          "|       GRAVITY NETWORK      |\n"
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
    experiment_ID, experiment_resume_ID = experimentID(typeID=parser.typeID,
                                                       parser=parser)

    # initialization
    path = initialization(network_name="GravityNet",
                          experiment_ID=experiment_ID,
                          experiment_resume_ID=experiment_resume_ID,
                          parser=parser,
                          debug=parser.debug_initialization)

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
    # $DATASET$
    if parser.dataset == '$DATASET$':
        dataset = DatasetName(images_dir=path['dataset']['images']['cropped'],
                              images_masks_dir=path['dataset']['images']['masks_cropped'],
                              annotations_dir=path['dataset']['annotations']['cropped'],
                              filename_list=data_split['filename'],
                              transforms=None)
    else:
        str_err = msg_error(file=__file__,
                            variable=parser.dataset,
                            type_variable="dataset",
                            choices="[$DATASET$]")
        sys.exit(str_err)

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
    num_images = dataset_num_images(dataset=parser.dataset,
                                    split=parser.split,
                                    do_dataset_augmentation=parser.do_dataset_augmentation,
                                    dataset_train=dataset_train,
                                    dataset_val=dataset_val,
                                    dataset_test=dataset_test)

    # ========================= #
    # DATASET NUM NORMAL IMAGES #
    # ========================= #
    # num normal images for dataset-train, dataset-val, dataset-test
    num_normal_images = dataset_num_normal_images(dataset=parser.dataset,
                                                  split=parser.split,
                                                  do_dataset_augmentation=parser.do_dataset_augmentation,
                                                  dataset_train=dataset_train,
                                                  dataset_val=dataset_val,
                                                  dataset_test=dataset_test)

    # ================== #
    # DATASET TRANSFORMS #
    # ================== #
    train_transforms, val_transforms, test_transforms = dataset_transforms(dataset=parser.dataset,
                                                                           normalization=parser.norm,
                                                                           parser=parser,
                                                                           statistics_path=path['dataset']['statistics'],
                                                                           debug=parser.debug_transforms)

    # apply dataset transforms
    dataset_train.dataset.transforms = train_transforms
    dataset_val.dataset.transforms = val_transforms
    dataset_test.dataset.transforms = test_transforms

    # ==================== #
    # DATASET AUGMENTATION #
    # ==================== #
    if parser.do_dataset_augmentation:
        # dataset-train augmented
        dataset_train = dataset_augmentation(dataset=parser.dataset,
                                             normalization=parser.norm,
                                             parser=parser,
                                             dataset_train=dataset_train,
                                             statistics_path=path['dataset']['statistics'],
                                             debug=parser.debug_transforms_augmentation)

    # ======================= #
    # DATASET NUM ANNOTATIONS #
    # ======================= #
    # num annotations for dataset-train, dataset-val, dataset-test
    num_annotations = dataset_num_annotations(dataset=parser.dataset,
                                              split=parser.split,
                                              do_dataset_augmentation=parser.do_dataset_augmentation,
                                              dataset_train=dataset_train,
                                              dataset_val=dataset_val,
                                              dataset_test=dataset_test)

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
    # image shape (H x W) -> after pre-processing (transforms)
    image_shape = np.array((int(parser.image_height), int(parser.image_width)))  # converts to numpy.array

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
                       image_shape=image_shape,
                       feature_map_shape=feature_map_shape,
                       num_gravity_points=num_gravity_points,
                       num_gravity_points_feature_map=num_gravity_points_feature_map)

    # debug execution
    debug_execution(do_debug_execution=parser.debug_execution)

    # =========== #
    # MODE: TRAIN #
    # =========== #
    if parser.mode in ['train', 'resume', 'train_test']:

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
                                device=device,
                                debug=parser.debug_hooking)

        # ==================== #
        # INIT METRICS (TRAIN) #
        # ==================== #
        metrics = metrics_dict(metrics_type='train')

        # training epochs range
        start_epoch_train = 1  # star train
        stop_epoch_train = start_epoch_train + parser.epochs  # stop train

        # ============ #
        # MODE: RESUME #
        # ============ #
        if parser.mode in ['resume']:
            # train epochs range
            start_epoch_train = parser.epoch_to_resume + 1  # star train
            stop_epoch_train = start_epoch_train + (parser.epochs - parser.epoch_to_resume)  # stop train

            # ================= #
            # LOAD RESUME MODEL #
            # ================= #
            print("\n------------------"
                  "\nLOAD RESUME MODEL:"
                  "\n------------------")

            # load resume model
            load_resume_model(net=net,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              path=path['model']['resume_to_load'])

            # ============== #
            # RESUME METRICS #
            # ============== #
            print("\n---------------"
                  "\nRESUME METRICS:"
                  "\n---------------")
            # resume metrics performance
            metrics = metrics_resume(metrics_resume_path=path['metrics']['resume'])

            # resume metrics-train.csv
            metrics_train_resume_csv(metrics_path=path['metrics']['train'],
                                     metrics=metrics)

            # =========== #
            # RESUME PLOT #
            # =========== #
            print("\n------------"
                  "\nRESUME PLOT:"
                  "\n------------")
            # resume FROC-validation
            resume_FROC_plot(experiment_ID=experiment_ID,
                             coords_resume_path=path['plots_validation']['resume']['coords_FROC'],
                             coords_path=path['plots_validation']['coords_FROC'],
                             plot_resume_path=path['plots_validation']['resume']['FROC'],
                             plot_path=path['plots_validation']['FROC'])

            # resume ROC-validation
            resume_ROC_plot(experiment_ID=experiment_ID,
                            coords_resume_path=path['plots_validation']['resume']['coords_ROC'],
                            coords_path=path['plots_validation']['coords_ROC'],
                            plot_resume_path=path['plots_validation']['resume']['ROC'],
                            plot_path=path['plots_validation']['ROC'])

            # resume output-gravity-validation
            resume_output_validation(experiment_ID=experiment_ID,
                                     output_resume_path=path['output']['resume']['gravity']['validation'],
                                     output_path=path['output']['gravity']['validation'])

        # for each epoch
        for epoch in range(start_epoch_train, stop_epoch_train):

            # ======== #
            # TRAINING #
            # ======== #
            print("\n---------"
                  "\nTRAINING:"
                  "\n---------")
            time_train_start = time.time()
            loss, classification_loss, regression_loss = train(dataset=parser.dataset,
                                                               num_epoch=epoch,
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
                       detections_path=path['detections']['validation'],
                       FP_list_path=select_FP_list_path(FP_images=parser.FP_images,
                                                        path=path['dataset']),
                       filename_output_gravity=select_output_gravity_filename(dataset=parser.dataset,
                                                                              split=parser.split),
                       output_gravity_path=path['output']['gravity']['validation'],
                       do_output_gravity=parser.do_output_gravity,
                       device=device,
                       debug=parser.debug_validation)
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
                                     TotalNumOfAnnotations=num_annotations['validation'],
                                     debug=parser.debug_FROC)

            # compute sensitivity
            sens_work_point_val, sens_max_val = sensitivity(FPS=FPS_val,
                                                            sens=sens_val,
                                                            work_point=parser.work_point)

            # compute ROC
            FPR_val, TPR_val = ROC(detections=detections_val)

            # compute AUFROC
            AUFROC_0_1_val = AUFROC(FPS=FPS_val, sens=sens_val, FPS_upper_bound=1)
            AUFROC_0_10_val = AUFROC(FPS=FPS_val, sens=sens_val, FPS_upper_bound=10)
            AUFROC_0_50_val = AUFROC(FPS=FPS_val, sens=sens_val, FPS_upper_bound=50)
            AUFROC_0_100_val = AUFROC(FPS=FPS_val, sens=sens_val, FPS_upper_bound=100)

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
            metrics['sensitivity']['work_point'].append(sens_work_point_val)
            metrics['sensitivity']['max'].append(sens_max_val)
            metrics['AUFROC']['[0, 1]'].append(AUFROC_0_1_val)
            metrics['AUFROC']['[0, 10]'].append(AUFROC_0_10_val)
            metrics['AUFROC']['[0, 50]'].append(AUFROC_0_50_val)
            metrics['AUFROC']['[0, 100]'].append(AUFROC_0_100_val)
            metrics['time']['train'].append(time_train)
            metrics['time']['validation'].append(time_val)
            metrics['time']['metrics'].append(time_metrics_val)

            # metrics-train.csv
            metrics_train_csv(metrics_path=path['metrics']['train'],
                              metrics=metrics)

            # show metrics train
            show_metrics_train(metrics=metrics,
                               work_point=parser.work_point)

            # =============== #
            # SAVE BEST MODEL #
            # =============== #
            print("\n----------------"
                  "\nSAVE BEST MODEL:"
                  "\n----------------")
            # save best-model with sensitivity work point
            if (epoch - 1) == np.argmax(metrics['sensitivity']['work_point']):
                save_best_model(epoch=epoch,
                                net=net,
                                metrics=metrics['sensitivity']['work_point'],
                                metrics_type='sensitivity work point',
                                optimizer=optimizer,
                                scheduler=scheduler,
                                path=path['model']['best']['sensitivity'])

            # save best-model with AUFROC [0, 10] metrics
            if (epoch - 1) == np.argmax(metrics['AUFROC']['[0, 10]']):
                save_best_model(epoch=epoch,
                                net=net,
                                metrics=metrics['AUFROC']['[0, 10]'],
                                metrics_type='AUFROC [0, 10]',
                                optimizer=optimizer,
                                scheduler=scheduler,
                                path=path['model']['best']['AUFROC'])

            # save resume-model
            save_resume_model(epoch=epoch,
                              net=net,
                              sensitivity_wp=metrics['sensitivity']['work_point'][-1],
                              AUFROC_0_10=metrics['AUFROC']['[0, 10]'][-1],
                              optimizer=optimizer,
                              scheduler=scheduler,
                              path=path['model']['resume'])

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
                             sensitivity_work_point=metrics['sensitivity']['work_point'],
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

        # check load model option
        check_load_model(parser=parser)

        # load best model sensitivity work point
        if parser.load_best_sensitivity_model:
            load_best_model(net=net,
                            metrics_type='sensitivity work point',
                            path=path['model']['best']['sensitivity'])

        # load best model AUFROC [0, 10]
        if parser.load_best_AUFROC_model:
            load_best_model(net=net,
                            metrics_type='AUFROC [0, 10]',
                            path=path['model']['best']['AUFROC'])

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
             detections_path=path['detections']['test'],
             FP_list_path=select_FP_list_path(FP_images=parser.FP_images,
                                              path=path['dataset']),
             output_gravity_path=path['output']['gravity']['test'],
             do_output_gravity=parser.do_output_gravity,
             device=device,
             debug=parser.debug_test)
        time_test = time.time() - time_test_start

        # read detections test for evaluation (numpy array)
        detections_test = read_csv(filepath_or_buffer=path['detections']['test'], usecols=["LABEL", "SCORE"]).dropna(subset='LABEL').values
        detections_score = detections_test[:, 1]  # detections score
        # detections_test_filename = path['detections']['test'].split('/')[3]
        # print("{} reading: COMPLETE".format(detections_test_filename))

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
                                   TotalNumOfAnnotations=num_annotations['test'],
                                   debug=parser.debug_FROC)

        # compute sensitivity
        sens_work_point_test, sens_max_test = sensitivity(FPS=FPS_test,
                                                          sens=sens_test,
                                                          work_point=parser.work_point)

        # compute AUFROC
        AUFROC_0_1_test = AUFROC(FPS=FPS_test, sens=sens_test, FPS_upper_bound=1)
        AUFROC_0_10_test = AUFROC(FPS=FPS_test, sens=sens_test, FPS_upper_bound=10)
        AUFROC_0_50_test = AUFROC(FPS=FPS_test, sens=sens_test, FPS_upper_bound=50)
        AUFROC_0_100_test = AUFROC(FPS=FPS_test, sens=sens_test, FPS_upper_bound=100)

        # compute ROC
        FPR_test, TPR_test = ROC(detections=detections_test)

        time_metrics_test = time.time() - time_metrics_test_start

        # update performance
        metrics['AUC'].append(AUC_test)
        metrics['sensitivity']['work_point'].append(sens_work_point_test)
        metrics['sensitivity']['max'].append(sens_max_test)
        metrics['AUFROC']['[0, 1]'].append(AUFROC_0_1_test)
        metrics['AUFROC']['[0, 10]'].append(AUFROC_0_10_test)
        metrics['AUFROC']['[0, 50]'].append(AUFROC_0_50_test)
        metrics['AUFROC']['[0, 100]'].append(AUFROC_0_100_test)
        metrics['time']['test'].append(time_test)
        metrics['time']['metrics'].append(time_metrics_test)

        # metrics-test.csv
        metrics_test_csv(metrics_path=path['metrics']['test'],
                         metrics=metrics)

        # show metrics test
        show_metrics_test(metrics=metrics,
                          work_point=parser.work_point)

        # ====== #
        # OUTPUT #
        # ====== #
        print("\n-------"
              "\nOUTPUT:"
              "\n-------")
        output(type_draw='box',
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

    # ============== #
    # MODE: TEST NMS #
    # ============== #
    if parser.mode in ['test_NMS']:

        # ======================= #
        # INIT METRICS (TEST NMS) #
        # ======================= #
        metrics = metrics_dict(metrics_type='test_NMS')

        # =============== #
        # LOAD BEST MODEL #
        # =============== #
        print("\n----------------"
              "\nLOAD BEST MODEL:"
              "\n----------------")

        # check load model option
        check_load_model(parser=parser)

        # load best model sensitivity work point
        if parser.load_best_sensitivity_model:
            load_best_model(net=net,
                            metrics_type='sensitivity work point',
                            path=path['model']['best']['sensitivity'])

        # load best model AUFROC [0, 10]
        if parser.load_best_AUFROC_model:
            load_best_model(net=net,
                            metrics_type='AUFROC [0, 10]',
                            path=path['model']['best']['AUFROC'])

        # ======== #
        # TEST NMS #
        # ======== #
        print("\n---------"
              "\nTEST NMS:"
              "\n---------")
        time_test_NMS_start = time.time()
        test_NMS(experiment_ID=experiment_ID,
                 net=net,
                 dataloader=dataloader_test,
                 hook=parser.hook,
                 gravity_points=gravity_points,
                 eval=parser.eval,
                 rescale_factor=parser.rescale,
                 NMS_box_radius=parser.NMS_box_radius,
                 detections_path=path['detections']['test_NMS'],
                 FP_list_path=select_FP_list_path(FP_images=parser.FP_images,
                                                  path=path['dataset']),
                 output_gravity_path=path['output']['gravity']['test_NMS'],
                 do_output_gravity=parser.do_output_gravity,
                 device=device,
                 debug=parser.debug_test)
        time_test_NMS = time.time() - time_test_NMS_start

        # read detections test for evaluation (numpy array)
        detections_test_NMS = read_csv(filepath_or_buffer=path['detections']['test_NMS'], usecols=["LABEL", "SCORE"]).dropna(subset=['LABEL']).values
        detections_NMS_score = detections_test_NMS[:, 1]
        # detections_test_filename = path['detections']['test'].split('/')[3]
        # print("{} reading: COMPLETE".format(detections_test_filename))

        # ================== #
        # METRICS (TEST NMS) #
        # ================== #
        time_metrics_test_NMS_start = time.time()

        # compute AUC
        AUC_test_NMS = AUC(detections=detections_test_NMS)

        # compute FROC
        FPS_test_NMS, sens_test_NMS = FROC(detections=detections_test_NMS,
                                           TotalNumOfImages=select_TotalNumOfImages(FP_images=parser.FP_images,
                                                                                    num_images=num_images['test'],
                                                                                    num_images_normals=num_normal_images['test']),
                                           TotalNumOfAnnotations=num_annotations['test'],
                                           debug=False)

        # compute sensitivity
        sens_work_point_test_NMS, sens_max_test_NMS = sensitivity(FPS=FPS_test_NMS,
                                                                  sens=sens_test_NMS,
                                                                  work_point=parser.work_point)

        # compute AUFROC
        AUFROC_0_1_test_NMS = AUFROC(FPS=FPS_test_NMS, sens=sens_test_NMS, FPS_upper_bound=1)
        AUFROC_0_10_test_NMS = AUFROC(FPS=FPS_test_NMS, sens=sens_test_NMS, FPS_upper_bound=10)
        AUFROC_0_50_test_NMS = AUFROC(FPS=FPS_test_NMS, sens=sens_test_NMS, FPS_upper_bound=50)
        AUFROC_0_100_test_NMS = AUFROC(FPS=FPS_test_NMS, sens=sens_test_NMS, FPS_upper_bound=100)

        # compute ROC
        FPR_test_NMS, TPR_test_NMS = ROC(detections=detections_test_NMS)

        time_metrics_test_NMS = time.time() - time_metrics_test_NMS_start

        # update performance
        metrics['AUC'].append(AUC_test_NMS)
        metrics['sensitivity']['work_point'].append(sens_work_point_test_NMS)
        metrics['sensitivity']['max'].append(sens_max_test_NMS)
        metrics['AUFROC']['[0, 1]'].append(AUFROC_0_1_test_NMS)
        metrics['AUFROC']['[0, 10]'].append(AUFROC_0_10_test_NMS)
        metrics['AUFROC']['[0, 50]'].append(AUFROC_0_50_test_NMS)
        metrics['AUFROC']['[0, 100]'].append(AUFROC_0_100_test_NMS)
        metrics['time']['NMS'].append(time_test_NMS)
        metrics['time']['metrics'].append(time_metrics_test_NMS)

        # metrics-test-NMS.csv
        metrics_test_NMS_csv(metrics_path=path['metrics']['test_NMS'],
                             metrics=metrics)

        # show metrics test NMS
        show_metrics_test_NMS(metrics=metrics,
                              NMS_type='{}x{}'.format(parser.NMS_box_radius, parser.NMS_box_radius),
                              work_point=parser.work_point)

        # ====== #
        # OUTPUT #
        # ====== #
        print("\n-------"
              "\nOUTPUT:"
              "\n-------")
        output(type_draw='box',
               dataset=dataset_test,
               num_images=parser.num_images,
               detections_path=path['detections']['test_NMS'],
               output_path=path['output']['test_NMS'],
               suffix="-output-NMS={}x{}|{}".format(parser.NMS_box_radius, parser.NMS_box_radius, experiment_ID))

        # ============= #
        # PLOT TEST NMS #
        # ============= #
        print("\n--------------"
              "\nPLOT TEST NMS:"
              "\n--------------")
        # FROC plot
        FROC_plot(title=plot_title['plots_test_NMS']['FROC'],
                  color='green',
                  experiment_ID=experiment_ID,
                  FPS=FPS_test_NMS,
                  sens=sens_test_NMS,
                  FROC_path=path['plots_test_NMS']['FROC'],
                  FROC_coords_path=path['plots_test_NMS']['coords']['FROC'])

        # FROC linear plot
        FROC_linear_plot(title=plot_title['plots_test_NMS']['FROC'],
                         color='green',
                         experiment_ID=experiment_ID,
                         FPS=FPS_test_NMS,
                         sens=sens_test_NMS,
                         FROC_upper_limit=10,
                         FROC_path=path['plots_test_NMS']['FROC_linear'])

        # ROC plot
        ROC_plot(title=plot_title['plots_test_NMS']['ROC'],
                 color='green',
                 experiment_ID=experiment_ID,
                 FPR=FPR_test_NMS,
                 TPR=TPR_test_NMS,
                 ROC_path=path['plots_test_NMS']['ROC'],
                 ROC_coords_path=path['plots_test_NMS']['coords']['ROC'])

        # Score Distribution
        score_distribution_plot(title=plot_title['plots_test_NMS']['score_distribution'],
                                score=detections_NMS_score,
                                bins=10000,
                                experiment_ID=experiment_ID,
                                score_distribution_path=path['plots_test_NMS']['score_distribution'])

    # execution mode complete
    execution_mode(mode=parser.mode,
                   option='complete')

if __name__ == "__main__":
    main()
