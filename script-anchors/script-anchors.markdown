# SCRIPT-ANCHORS

**NOTE**: run from the repository root with `PYTHONPATH=.` so the `net` package resolves.

## 🌌 GravityPoints-Configuration.py

Save gravity-points configuration

    PYTHONPATH=. python3 script-anchors/GravityPoints-Configuration.py script_anchors \
        --image_height 3328 \
        --image_width 2560 \
        --config grid-10 \
        --save_config

## 🌌 GravityPoints-Hooking.py

Save gravity-points hooking process

    PYTHONPATH=. python3 script-anchors/GravityPoints-Hooking.py script_anchors \
        --dataset_path "path to dataset main folder" \
        --images_extension tiff \
        --dataset "Dataset Name" \
        --idx 0 \
        --rescale 1.0 \
        --config grid-10 \
        --hook 10
