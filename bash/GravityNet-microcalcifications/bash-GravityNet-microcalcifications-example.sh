#!/bin/bash

#  | ================================================= |
#  |      GRAVITY NETWORK for microcalcifications      |
#  |                   DATASET CHECK                   |
#  | ================================================= |

# ---------- #
# PARAMETERS #
# ---------- #
# dataset=INbreast
# do_dataset_augmentation
# rescale=1.0
# norm=none
# epochs=1
# lr=1e-04
# bs=8
# backbone=ResNet-34
# pretrained
# config=grid-10
# hook=10
# eval=distance7
# num_images=205

# CUDA_VISIBLE_DEVICES=3


# ------------------ #
# EXPERIMENT EXAMPLE #
# ------------------ #
# train (1-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py train --where=data --dataset=INbreast --do_dataset_augmentation --split=1-fold --rescale=1.0 --norm=none --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-34 --pretrained --config=grid-10 --hook=10 --eval=distance7 --do_output_gravity > train-example-1fold.txt
# train (2-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py train --where=data --dataset=INbreast --do_dataset_augmentation --split=2-fold --rescale=1.0 --norm=none --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-34 --pretrained --config=grid-10 --hook=10 --eval=distance7 --do_output_gravity > train-example-2fold.txt

# test (1-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py test --where=data --dataset=INbreast --do_dataset_augmentation --split=1-fold --rescale=1.0 --norm=none --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-34 --pretrained --config=grid-10 --hook=10 --eval=distance7 --do_output_gravity --num_images=205 > test-example-1fold.txt
# test (2-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py test --where=data --dataset=INbreast --do_dataset_augmentation --split=2-fold --rescale=1.0 --norm=none --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-34 --pretrained --config=grid-10 --hook=10 --eval=distance7 --do_output_gravity --num_images=205 > test-example-2fold.txt

# test NMS (1-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py test_NMS --where=data --NMS_box_radius=7 --dataset=INbreast --do_dataset_augmentation --split=1-fold --rescale=1.0 --norm=none --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-34 --pretrained --config=grid-10 --hook=10 --eval=distance7 --do_output_gravity --num_images=205 > test-example-NMS-7x7-1fold.txt
# test NMS (2-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py test_NMS --where=data --NMS_box_radius=7 --dataset=INbreast --do_dataset_augmentation --split=2-fold --rescale=1.0 --norm=none --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-34 --pretrained --config=grid-10 --hook=10 --eval=distance7 --do_output_gravity --num_images=205 > test-example-NMS-7x7-2fold.txt
