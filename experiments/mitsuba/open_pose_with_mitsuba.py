# Simple inverse rendering example: render a cornell box reference image,
# then replace one of the scene parameters and try to recover it using
# differentiable rendering and gradient-based optimization. (PyTorch)
import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

import time
import torch
import cv2

from mitsuba.python.autodiff import render, render_torch, write_bitmap
from mitsuba.python.util import traverse
from mitsuba.core.xml import load_file
from mitsuba.core import Thread, Vector3f


from torch_openpose.body import Body
from torch_openpose import util

body_estimation = Body(
    '/home/ax/data/programs/pytorch-openpose/model/body_pose_model.pth', True)

Thread.thread().file_resolver().append('pose_scene')
scene = load_file('pose_scene/scene.xml')

# Find differentiable scene parameters
#params = traverse(scene)

# print(params)

# Render a reference image (no derivatives used yet)
image_ref = render(scene, spp=8)
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap('pose_scene/out_ref.png', image_ref, crop_size)

# Render a reference image (no derivatives used yet)
rendering_torch = render_torch(scene, spp=8)
print(rendering_torch.shape)
print(torch.min(rendering_torch))
print(torch.max(rendering_torch))
