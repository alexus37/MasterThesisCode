# Simple inverse rendering example: render a cornell box reference image,
# then replace one of the scene parameters and try to recover it using
# differentiable rendering and gradient-based optimization. (PyTorch)
import time
import torch
import cv2
import copy
from matplotlib import pyplot as plt

from torch_openpose.body import Body
from torch_openpose import util


def read_imgfile(path, width=None, height=None, image_type=cv2.IMREAD_COLOR):
    val_image = cv2.imread(path, image_type)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


body_estimation = Body(
    '/home/ax/data/programs/pytorch-openpose/model/body_pose_model.pth', True)

IMAGE_WIDTH, IMAGE_HEIGHT = 432, 368
image_path_target = 'pose_scene/out_ref.png'
image_target = read_imgfile(
    image_path_target, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

candidate, subset = body_estimation(image_target)
canvas = copy.deepcopy(image_target)
canvas = util.draw_bodypose(canvas, candidate, subset)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(canvas[:, :, [2, 1, 0]])
ax.axis('off')
plt.show()
