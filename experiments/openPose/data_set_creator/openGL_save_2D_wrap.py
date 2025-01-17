import numpy as np
from PIL import Image, ImageOps
from pyrr import Matrix44, Vector4, Vector3, matrix44
import cv2

import argparse
import os
import xml.dom.minidom
import glob
from tqdm import tqdm

VERTICES = np.array([[1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [-1.0, -1.0, 0.0],
                    [-1.0, 1.0, 0.0]],
                        dtype="float32")
# NORI image size
#WINDOW_WIDTH, WINDOW_HEIGHT = 768, 576
WINDOW_WIDTH, WINDOW_HEIGHT = 432, 368

NOISE_WIDTH, NOISE_HEIGHT = 200, 400
# images used by the network
w, h = 432, 368

# camera params
FAR_CLIP = 2500.0
NEAR_CLIP = 2.0
FOV = 45.0

ORIGIN = np.array([-4.21425, 105.008, 327.119], dtype="float32")
TARGET = np.array([-4.1969, 104.951, 326.12], dtype="float32")
UP = np.array([0.0, 1.0, 0.0], dtype="float32")

# RECT PARAMS
RECT_SCALE = Vector3([15.0, 30.0, 1.0])
RECT_TRANSLATE = Vector3([0.0, 110.0, 15.0])

class Save2dWarp:
    def __init__(self, x = 50.0, y=0.0, z =-50, angle=1.5):
        self.mvp_matrix = np.transpose(Save2dWarp.compute_mvp(Vector3([x, y, z]), angle))
    @staticmethod
    def compute_mvp(translation, rotation):
        # model matrix is correct
        identity_matrix = np.identity(4)
        scale_matrix = np.transpose(matrix44.create_from_scale(RECT_SCALE))
        trans_matrix = np.transpose(matrix44.create_from_translation(RECT_TRANSLATE))
        rot_matrix = np.transpose(matrix44.create_from_y_rotation(np.radians(360.0 - rotation)))
        trans_matrix_cur = np.transpose(matrix44.create_from_translation(translation))

        model_matrix = identity_matrix
        model_matrix = np.matmul(model_matrix, trans_matrix_cur)
        model_matrix = np.matmul(model_matrix, rot_matrix)
        model_matrix = np.matmul(model_matrix, trans_matrix)
        model_matrix = np.matmul(model_matrix, scale_matrix)


        view_matrix = np.transpose(
            matrix44.create_look_at(
                ORIGIN,
                TARGET,
                UP
            )
        )

        proj_matrix = np.transpose(
            matrix44.create_perspective_projection(
                FOV,
                WINDOW_WIDTH / WINDOW_HEIGHT,
                NEAR_CLIP,
                FAR_CLIP
            )
        )
        cam_matrix = np.matmul(proj_matrix, view_matrix)
        m = np.matmul(cam_matrix, model_matrix)

        return np.transpose(m)

    @staticmethod
    def order_points(pts):
    	# initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect

    @staticmethod
    def save_warp_npy(warp, filename):
        # Transform to record for tf
        tf_foramt = np.array(warp.flatten()[:-1], dtype=np.float32)
        np.save(filename, tf_foramt)


    def compute_pixel_coordinates(self):
        pixel_coordinates = []

        for vertex in VERTICES:
            homo_vertex = np.append(vertex, 1.0)
            # Transform the face into clip space.
            vertex_clip_space = np.matmul(self.mvp_matrix, homo_vertex)

            #  Apply perspective division.
            vertex_normalize_device_space = np.array([
                vertex_clip_space[0] / vertex_clip_space[3],
                vertex_clip_space[1] / vertex_clip_space[3],
                vertex_clip_space[2] / vertex_clip_space[3],
                vertex_clip_space[3]]
            )

            # Transform the face into screen space.
            vertex_screen_space = np.array([
                np.floor(0.5 * WINDOW_WIDTH * (vertex_normalize_device_space[0] + 1.0)),
                # flip image
                WINDOW_HEIGHT - np.floor(0.5 * WINDOW_HEIGHT * (vertex_normalize_device_space[1] + 1.0)),
                vertex_normalize_device_space[2],
                vertex_normalize_device_space[3]
            ])


            pixel_coordinates.append(vertex_screen_space[:2])
        return pixel_coordinates

    @staticmethod
    def compute_wrap_matrix(pixel_coordinates, maxHeight, maxWidth):
        src = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")
        dst = Save2dWarp.order_points(np.array(pixel_coordinates))
        # compute the perspective transform matrix and then apply it
        # transflow needs it in that order cv2 the other way around
        return cv2.getPerspectiveTransform(dst, src)

def get_scene_values_from_filename(file_name):
    doc = xml.dom.minidom.parse(file_name)

    shape_node_rect = doc.getElementsByTagName('shape')[1]
    transform_node = shape_node_rect.getElementsByTagName('transform')[0]
    translation_node = transform_node.getElementsByTagName('translate')[1]
    rotation_node = transform_node.getElementsByTagName('rotate')[0]

    x = float(translation_node.getAttribute('x'))
    y = float(translation_node.getAttribute('y'))
    z = float(translation_node.getAttribute('z'))

    angle = float(rotation_node.getAttribute('angle'))
    return x, y, z, angle

def main(file_name):
    if not os.path.isfile(file_name):
        print('File does not exist')
        return

    [x, y, z, angle] = get_scene_values_from_filename(file_name)
    print(f'using: x={x}, y={y}, z={z}, angle={angle}')

    plain_file_name, _ = os.path.splitext(file_name)

    gl = Save2dWarp(x, y, z, angle)

    # get the pixel coordinates
    pixel_coordinates = gl.compute_pixel_coordinates()

    # transform to warp
    warp = Save2dWarp.compute_wrap_matrix(pixel_coordinates, NOISE_HEIGHT, NOISE_WIDTH)

    # save to npy
    Save2dWarp.save_warp_npy(warp, f"{plain_file_name}_warp.npy")
def main_multiple(directory):
    for file_name in tqdm(glob.glob(f"{directory}/train/*.xml")):
        [x, y, z, angle] = get_scene_values_from_filename(file_name)
        plain_file_name, _ = os.path.splitext(file_name)
        gl = Save2dWarp(x, y, z, angle)
        # get the pixel coordinates
        pixel_coordinates = gl.compute_pixel_coordinates()
        # transform to warp
        warp = Save2dWarp.compute_wrap_matrix(pixel_coordinates, NOISE_HEIGHT, NOISE_WIDTH)
        # save to npy
        Save2dWarp.save_warp_npy(warp, f"{plain_file_name}_warp.npy")

    for file_name in tqdm(glob.glob(f"{directory}/test/*.xml")):
        [x, y, z, angle] = get_scene_values_from_filename(file_name)
        plain_file_name, _ = os.path.splitext(file_name)
        gl = Save2dWarp(x, y, z, angle)
        # get the pixel coordinates
        pixel_coordinates = gl.compute_pixel_coordinates()
        # transform to warp
        warp = Save2dWarp.compute_wrap_matrix(pixel_coordinates, NOISE_HEIGHT, NOISE_WIDTH)
        # save to npy
        Save2dWarp.save_warp_npy(warp, f"{plain_file_name}_warp.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute mask from scenefile')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--filename', type=str, help='filename of the xml file')
    group.add_argument('--directory', type=str, help='directory where train and test exist with the scene xmls')
    args = parser.parse_args()

    if args.filename is not None:
        main(args.filename)

    if args.directory is not None:
        main_multiple(args.directory)
