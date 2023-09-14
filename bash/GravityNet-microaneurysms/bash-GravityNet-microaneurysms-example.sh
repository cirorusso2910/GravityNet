#!/bin/bash

#  | ============================================ |
#  |      GRAVITY NETWORK for microaneurysms      |
#  |                    EXAMPLE                   |
#  | ============================================ |

# ---------- #
# PARAMETERS #
# ---------- #
# dataset=E-ophtha-MA
# do_dataset_augmentation
# norm=min-max
# channel=G
# epochs=1
# lr=1e-04
# bs=8
# backbone=ResNet-152
# pretrained
# config=grid-10
# hook=10
# eval=radius1
# num_images=189 (1-fold)
# num_images=192 (2-fold)

# CUDA_VISIBLE_DEVICES=3


# ------------------ #
# EXPERIMENT EXAMPLE #
# ------------------ #
# train (1-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py train --where=home --dataset=E-ophtha-MA --do_dataset_augmentation --split=1-fold --norm=min-max --channel=G --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-152 --pretrained --config=grid-10 --hook=10 --eval=radius1 --do_output_gravity > train-example-1fold.txt
# train (2-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py train --where=home --dataset=E-ophtha-MA --do_dataset_augmentation --split=2-fold --norm=min-max --channel=G --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-152 --pretrained --config=grid-10 --hook=10 --eval=radius1 --do_output_gravity > train-example-2fold.txt

# test (1-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py test --where=home --dataset=E-ophtha-MA --do_dataset_augmentation --split=1-fold --norm=min-max --channel=G --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-152 --config=grid-10 --hook=10 --eval=radius1 --do_output_gravity --num_images=189 > test-example-1fold.txt
# test (2-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py test --where=home --dataset=E-ophtha-MA --do_dataset_augmentation --split=2-fold --norm=min-max --channel=G --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-152 --config=grid-10 --hook=10 --eval=radius1 --do_output_gravity --num_images=192 > test-example-2fold.txt

# test NMS (1-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py test_NMS --where=home --NMS_box_radius=3 --do_dataset_augmentation --dataset=E-ophtha-MA --split=1-fold --norm=min-max --channel=G --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-152 --config=grid-10 --hook=10 --eval=radius1 --do_output_gravity --num_images=189 > test-example-NMS-3x3-1fold.txt
# test NMS (2-fold)
#CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py test_NMS --where=home --NMS_box_radius=3 --do_dataset_augmentation --dataset=E-ophtha-MA --split=2-fold --norm=min-max --channel=G --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-152 --config=grid-10 --hook=10 --eval=radius1 --do_output_gravity --num_images=192 > test-example-NMS-3x3-2fold.txt
