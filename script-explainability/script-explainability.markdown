# SCRIPT-EXPLAINABILITY

**NOTE**: run from the repository root with `PYTHONPATH=.` so the `net` package resolves.

## 🔦 Explainability-GradCAM.py

Apply Explainable AI Grad-CAM method

    PYTHONPATH=. python3 script-explainability/Explainability-GradCAM.py explainability \
        --dataset_path "path to dataset main folder" \
        --experiments_path "path to experiments result" \
        --images_extension tiff \
        --images_masks_extension png \
        --annotations_extension csv \
        --load_best_AUFROC_0_10_model \
        --dataset "Dataset Name" \
        --do_dataset_augmentation \
        --num_channels 1 \
        --split 1-fold \
        --rescale 1.0 \
        --norm none \
        --epochs 100 \
        --lr 1e-04 \
        --bs 8 \
        --backbone ResNet-152 \
        --config grid-15 \
        --hook 15 \
        --eval distance10 \
        --GPU "GPU device name" \
        --do_NMS \
        --NMS_box_radius "Lesion Radius" \
        --num_images "Num Images"
