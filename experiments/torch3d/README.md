# Torch3d

This directory contains all experiments with the PyTorch3D renderer.

## 01_render_image.ipynb

Render an image using the PyTorch3d renderer and compute vertex attribution for a joint.

## 011_render_image_multiple_viewports.ipynb

Render an image from multiple view points and average gradients for attribution.

## 02_mts_debug.ipynb

Verify that scene files are correctly rendered in PyTorch3D by comparing with Mitsuba.

## 02_render_optimize_texture.ipynb

Use PyTorch3D to optimize for texture.

## 03_render_2_spheres.ipynb

Use PyTorch3D to render 2 spheres with different color

## 04_optimize_single_tshirt.ipynb

Optimize for a single T-shirt texture to fool openPose

## 041_optimize_single_tshirt_bg.ipynb

Optimize for a single T-shirt with a background texture to fool openPose

## 042_render_video.ipynb

Render a video for every frame of a MoCap sequence.

## 043_render_radial_bar_chart.ipynb

Analyse the T-shirt texture for a MoCap sequence from different directions.

## 044_render_on_real_image.ipynb

Render the T-shirt with the universal texture on top of a real soldier image

## 045_render_on_real_video.ipynb

Render the T-shirt with the universal texture on top of a real image

## 05_optimize_multiple_tshirt.ipynb

Optimize for a universal texture for multiple images.

## 051_optimize_multiple_tshirt_bg.ipynb

Optimize for a universal texture for multiple images with backgrounds

## 052_optimize_multiple_tshirt_bg_Laplacian.ipynb

Optimize for a universal texture for multiple images with backgrounds using a laplacian color regularization

## 053_optimize_multiple_tshirt_bg_kl_loss.ipynb

Optimize for a universal texture for multiple images with backgrounds with the kl divergence loss

## 054_optimize_multiple_tshirt_bg_kl_loss_targeted.ipynb

Optimize for a universal texture for multiple images with backgrounds with the kl divergence loss target for certain joints

## 06_optimize_single_tshirt_mesh.ipynb

Optimize for a single T-shirt geometry

## 061_optimize_single_tshirt_texture.ipynb

Optimize for a single T-shirt geometry with a regularizer

## 07_optimize_pose.ipynb

Try to optimize the pose of a mesh by comparing the confidence maps and pafs of a target state.

## 08_optimize_multiple_pose.ipynb

Try to optimize the pose of a mesh by comparing the confidence maps and pafs of a target state,
but use different view points.

## 09_SMPL_attribution.ipynb

Connect PyTorch3D with SMPL and compute attribution for the SMPL attributes

## 10_SMPL_optmize_pose_image.ipynb

Try to optimize SMPL parameters for a target state.

## 101_SMPL_optmize_pose_from_SMPL.ipynb

Try to optimize SMPL parameters for a target state with a large change.

## 102_SMPL_optmize_pose_from_SMPL_mini_change.ipynb

Try to optimize SMPL parameters for a target state with a small change.

## 103_SMPL_optmize_pose_from_SMPL_rendering.ipynb

Try to optimize SMPL parameters for a target by comparing only the renderings.

## 104_SMPL_optmize_pose_from_direct_vertex_positions.ipynb

Try to optimize SMPL parameters for a target by comparing directly the vertex positions.

## 11_SMPL_robustness_beta.ipynb

Test the robustness of openPose by varying the beta parameters of the SMPL model

## 12_SMPL_robustness_pose.ipynb

Test the robustness of openPose by varying the position parameters of the SMPL model

## analyse_runs.ipynb

Analyse different runs.
