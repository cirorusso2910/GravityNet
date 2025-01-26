# SCRIPT-EXPLAINABILITY

## :flashlight: Explainability-GradCAM.py

Apply Explainable AI Grad-CAM method

    Explainability-GradCAM.py explainability
        --dataset_path              =       "path to dataset main folder"
        --experiments_path          =       "path to experiments result"
        --images_extension          =       [png, tif]
        --images_masks_extension    =       [png, none]
        --annotations_extension     =       csv
        --load_best_[]_model [AUFROC, AUPR, sensitivity-10-FPS] 
        --dataset                   =       "Dataset Name"
        --do_dataset_augmentation
        --num_channels              =       [1, 3]
        --split                     =       [default, 1-fold, 2-fold]
        --rescale                   =       [0,5, 1.0]
        --norm                      =       [none, min-max, std] 
        --epochs                    =       100
        --lr                        =       1e-04
        --bs                        =       8
        --backbone                  =       ResNet-152
        --config                    =       grid-15
        --hook                      =       15
        --eval                      =       distance10
        --GPU                       =       "GPU device name"
        --do_NMS                    
        --NMS_box_radius            =       "Lesion Radius"
        --num_images                =       "Num Images"
