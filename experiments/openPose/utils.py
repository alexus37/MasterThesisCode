from skimage import feature, transform
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import IFrame
import warnings
import http.server
import socketserver
import asyncio
import websockets
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common
import cv2
import xml.dom.minidom
import os 
import glob 
import random
from PIL import Image
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
from deepexplain.tf.v1_x import DeepExplain
from tf_pose.common import CocoPart, read_imgfile


def run_websocket_server(websocket_handler, port=1234):
    print(f'Starting websocket server on port {port}')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(websocket_handler, "127.0.0.1", port)
    loop.run_until_complete(start_server)
    loop.run_forever()

def run_http_server(port = 8080):
    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print("serving at port", port)
        httpd.serve_forever()


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def compare_images(img1, img2):
    if not is_port_in_use(8000):
        warnings.warn('The server seems not to be running pls start with: python -m http.server')
        return

    # save the images
    abs_max = np.percentile(np.abs(img1), 100)
    abs_min = abs_max
    plt.imsave('./html/first.png', img1, cmap='Greys', vmin=-abs_min, vmax=abs_max)

    colormap_transparent = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['blue','red'], 256)
    colormap_transparent._init() # create the _lut array, with rgba values

    alphas = np.linspace(0, 2.0 * np.pi, colormap_transparent.N+3)
    colormap_transparent._lut[:,-1] = list(map(lambda x : np.clip(np.cos(x) + 0.5, 0.0, 1.0), alphas))

    plt.imsave('./html/second.png', img2, cmap=colormap_transparent)

    return IFrame(src='http://0.0.0.0:8000/html/diffViewer.html', width=700, height=600)


def get_humans_as_lines(humans, image_h, image_w):
    centers = {}
    lines = []
    for human in humans:
        # draw point
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), image_h - int(body_part.y * image_h + 0.5))
            centers[i] = center

        # draw line
        for pair_order, pair in enumerate(common.CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue


            line = [centers[pair[0]], centers[pair[1]]]
            lines.append(line)
    return lines


def is_same_image(a, b):
    return np.sum(np.abs(a-b))

def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 4.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

def print_image_Stats(img):
    print(f'Min: {np.min(img)}')
    print(f'Max: {np.max(img)}')
    print(f'Avg: {np.average(img)}')

def compare_poses(pose1, pose2):
    total_diff = 0
    cur_max = 0
    most_moved_part = ''
    if(len(pose1.body_parts) != len(pose2.body_parts)):
        print('Poses have different length of body parts')
        return 0, 'NONE'
    for i in pose1.body_parts.keys():
        part1 = pose1.body_parts[i]
        part2 = pose2.body_parts[i]
        part1_pos =  np.array([part1.x, part1.y])
        part2_pos =  np.array([part2.x, part2.y])
        cur_diff = np.linalg.norm(part1_pos - part2_pos)
        if cur_diff > cur_max:
            most_moved_part = part1.get_part_name()
            cur_max = cur_diff
        total_diff += cur_diff
    return total_diff, most_moved_part
    

def pose_loss_single_human(newHuman, oldHuman):
    if len(oldHuman) == 0 or len(newHuman) == 0:
        return 0
    detected = 0
    for i in oldHuman[0].body_parts.keys():
        if i in newHuman[0].body_parts:
            detected += 1
    return detected / len(oldHuman[0].body_parts)

def pose_loss_single_human_per_part(newHuman, oldHuman):
    if len(oldHuman) == 0 or len(newHuman) == 0:
        return {}
    result = {}
    for i in oldHuman[0].body_parts.keys():
        # the joint existed before
        result[i] = 0
        # the joint is successfully not detected after
        if i in newHuman[0].body_parts:
            result[i] = 1
    return result

def compute_distance(pose1, pose2, index):
    if index not in pose1.body_parts or index not in pose2.body_parts:
        print(f'{index} not in both poses found')
        return 0
    part1 = pose1.body_parts[index]
    part2 = pose2.body_parts[index]
    part1_pos =  np.array([part1.x, part1.y])
    part2_pos =  np.array([part2.x, part2.y])
    return np.linalg.norm(part1_pos - part2_pos)


def load_batch(training_paths, start_index, batch_size, width=432, height=368):
    i = start_index
    batch = []
    while len(batch) < batch_size:
        if i >= len(training_paths):
            i = 0
        batch.append(common.read_imgfile(str(training_paths[i]), width, height))
        i += 1
    return np.asarray(batch), i

    
    
def open_pose_attribution(estimator, 
                          image, 
                          method='grad*input', 
                          Y_shape = [None] + [1, 93, 93], 
                          QUANTILE=0.99,
                          heatmap=None, c_id=CocoPart.RShoulder.value,
                          mask_of_mask=None):
    with DeepExplain(session=estimator.persistent_sess, graph=estimator.graph) as de:
        input_tensor = estimator.tensor_image
        output_tensor = estimator.tensor_heatMat[:, :, :, c_id]
        xs = [image]
        
        if heatmap is not None:
            quant = np.quantile(heatmap, QUANTILE)
            mask = heatmap > quant
            mask = Image.fromarray(np.uint8(mask*255))
            mask = np.array(mask.resize(Y_shape[2:], Image.ANTIALIAS))
            if mask_of_mask is not None:
                mask = mask * (mask_of_mask > 0)
            ys = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)
        else:
            ys = np.ones([1, ] + Y_shape[1:])
        [attr] = de.explain(method, T=output_tensor, X=input_tensor, xs=xs, ys=ys, Y_shape=Y_shape)
        return attr, ys
    
    
def get_image_transform_mask(directory, IMAGE_WIDTH, IMAGE_HEIGHT):
    image_path_targets = []
    image_targets = []
    image_masks = []
    image_transforms = []
    for file in glob.glob(f"{directory}/*.xml"):
        filename, file_extension = os.path.splitext(file)
        
        image_path_targets.append(filename)
        
        image_targets.append(read_imgfile(f'{filename}.jpg', IMAGE_WIDTH, IMAGE_HEIGHT))
        
        mask = read_imgfile(f'{filename}_mask.png', IMAGE_WIDTH, IMAGE_HEIGHT, cv2.IMREAD_GRAYSCALE) > 0
        image_masks.append(np.array(np.repeat(mask[:, :, np.newaxis], 3, 2)))
        
        image_transforms.append(np.load(f'{filename}_warp.npy') )
    
    merged_list = list(zip(image_path_targets, image_targets, image_masks, image_transforms))
    random.seed(42)
    random.shuffle(merged_list)
    image_path_targets, image_targets, image_masks, image_transforms = zip(*merged_list)
    
    return image_path_targets, image_targets, image_masks, image_transforms
