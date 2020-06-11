# OpenPose

This directory contains experiments with openPose in Tensorflow as well as PyTorch.

## 00_Estimation_Online.ipynb

Example notebook how to connect a notebook to a website

## 00_WebsocketTest.ipynb

Example notebook how to connect a notebook to a website using a websocket.

## 01_Understand.ipynb

Understand the internals of openPose.

## 02_Understand_PAF.ipynb

Understand the part affinity fields used in openPose.

## 03_Understand_Loss.ipynb

Understand the loss function used in openPose.

## 04_Multiple_Images.ipynb

Run attribution on multiple images on the least confident confidence map

## 05_Manipulate_Wrist.ipynb

Changes the values of the confidence maps and paf to manipulate the wrist position.

## 06_Manipulate_Joints.ipynb

Changes the values of the confidence maps and paf to manipulate the joint position.

## 07_Pose_Adv.ipynb

Change the pixels of the image to get a required pose.

## 08_Adv_Verify_Peak_Moves.ipynb

Verify how the confidence maps change during the optimization process.

## 09_Adv_Target_Mask.ipynb

Change the pixels of the image inside a masked region to get a required pose.

## 10_Adv_Universal.ipynb

Compute a universal noise pattern for a targeted pose.

## 11_Adv_Universal_Attribution.ipynb

Compute the attribution of the universal noise pattern

## 12_Adv_Universal_Transform.ipynb

Check how the universal noise pattern behaves if it is transformed
(scaled, rotated, translated).

## 13_Adv_Universal_Invisible.ipynb

Compute a universal noise pattern to be invisible.

## 14_Adv_Universal_Different_Poses.ipynb

Compute universal noise pattern with a different target pose.

## 15_Adv_Universal_on_rendering.ipynb

Compute poster texture to fool single image using a mask.

## 16_Adv_on_rendering_with_2D_warp.ipynb

Compute poster texture to fool multiple single images using 2D warp function.

## 17_Adv_Universal_on_rendering_with_2D_warp_main.ipynb

Compute universal noise poster texture.

## 18_Adv_Universal_torch.ipynb

Verify equality of PyTorch and Tensorflow openPose implementation

## analyse_runs.ipynb

Analyse different runs with different loss functions and regulations.
