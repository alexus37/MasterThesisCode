# Mitsuba

This directory includes all experiments with the differentiable renderer [Mitsuba](https://github.com/mitsuba-renderer/mitsuba2).

## 01_open_pose_test.ipynb

The Notebook that tests the pose detection on renderer data from the [DeepGarment](https://cgl.ethz.ch/Downloads/Publications/Papers/2017/Dib17a/Dib17a.pdf)
dataset.

## 02_mts_openPose.ipynb

A notebook that tests the connection between Mitsuba and openPose

## 03_mts_grad_vertex.ipynb

A notebook that computes vertex attribution using Mitsuba.

## 04_mts_grad_texture.ipynb

A notebook that computes gradients with respect to texture.

## 041_mts_get_texture_tv.ipynb

A notebook that optimizes poster texture with a regulation on total variation.

## 042_mts_get_texture_blur.ipynb

A notebook that optimizes poster texture with a regulation on sharpness.

## 043_mts_get_texture_blur_with_loss.ipynb

A notebook that optimizes poster texture with a regulation on blurriness of the texture.

## 044_mts_debug_tonemap.ipynb

A notebook that implements tone mapping as a differentiable operation

## 05_mts_grad_texture_multiple_scenes.ipynb

A notebook that optimizes for "invisible" texture for a poster with multiple training scenes

## 06_mts_grad_texture_multiple_scenes_tv.ipynb

A notebook that optimizes for "invisible" texture for a poster with multiple training scenes and a total variation regularizer.

## 07_tshirt_single.ipynb

A notebook that optimizes for "invisible" texture for a poster with a single training scene

## 071_tshirt_single_point_light.ipynb

A notebook that optimizes for "invisible" texture for a poster with a single training scene with a point light.

## 08_tshirt.ipynb

A notebook that optimizes for "invisible" T-Shirt texture with a single training scene

## 081_tshirt_pointlight.ipynb

A notebook that optimizes for "invisible" T-Shirt texture with a single training scene and a point light
