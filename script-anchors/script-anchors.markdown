# SCRIPT-ANCHORS

## 🌌 GravityPoints-Configuration.py

Save gravity-points configuration

    GravityPoints-Configuration.py script_anchors
        --image_height              =       image height (H)
        --image_width               =       image widht (W)
        --config                    =       [grid, dice]
        --save_config

## 🌌 GravityPoints-Hooking.py

Save gravity-points hooking process

    GravityPoints-Hooking.py script_anchors
        --dataset_path              =       "path to dataset main folder"
        --images_extension          =       [png, tif]
        --dataset                   =       "Dataset Name"
        --idx                       =       dataset sample index
        --rescale                   =       [0,5, 1.0]
        --config                    =       [grid, dice]
        --hook                      =       hooking distance
