import xml.dom.minidom
import torch
import os
import torch
import xml.dom.minidom
from pyrr import Vector3
import pyrr

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, 
    BlendParams,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturedSoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader
)
from pytorch3d.renderer.cameras import look_at_view_transform

from torch_openpose import util
import numpy as np

import cv2
import copy

from tqdm import tqdm
import glob
import random
# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

def where(cond, x_1, x_2):
    cond = cond.float()    
    return (cond * x_1) + ((1-cond) * x_2)

def get_scenes_filenames(directory, include_filter=[], lim=10000000):
    scenes = []
    
    for file in tqdm(glob.glob(f"{directory}/*.xml")):
        if len(include_filter) == 0:
            scenes.append(file)
        else:
            for mesh_name in include_filter:
                if mesh_name in file:
                    scenes.append(file)
                    break
        
        if len(scenes) > lim:
            break
    # shuffel scenes due to naming 
    random.seed(42)
    random.shuffle(scenes)
    
    return scenes

def get_background_filenames(directory, include_filter=[], lim=10000000):
    bg = []
    
    for file in glob.glob(f"{directory}/*.jpg"):
        if len(include_filter) == 0:
            bg.append(file)
        else:
            for mesh_name in include_filter:
                if mesh_name in file:
                    bg.append(file)
                    break
        
        if len(bg) > lim:
            break
    # shuffel scenes due to naming 
    random.shuffle(bg)
    
    return bg


def transform_node_to_matrix(node):
    model_matrix = np.identity(4)
    for child in reversed(node.childNodes):
        if child.nodeName == "translate":
            x = float(child.getAttribute('x'))
            y = float(child.getAttribute('y'))
            z = float(child.getAttribute('z'))
            z *= -1
            translate_vec = Vector3([x, y, z])
            trans_matrix = np.transpose(pyrr.matrix44.create_from_translation(translate_vec))
            model_matrix = np.matmul(model_matrix, trans_matrix)
        if child.nodeName == "scale":
            scale = float(child.getAttribute('value'))
            scale_vec = Vector3([scale, scale, scale])
            scale_matrix = np.transpose(pyrr.matrix44.create_from_scale(scale_vec))
            model_matrix = np.matmul(model_matrix, scale_matrix)
            
    return model_matrix

def transform_node_to_R_T(node):
    eye = node.getAttribute('origin').split(',')
    eye = [float(i) for i in eye]
    at = node.getAttribute('target').split(',')
    at = [float(i) for i in at]
    up = node.getAttribute('up').split(',')
    up = [float(i) for i in up]
    
    R, T = look_at_view_transform(
        eye=[eye], 
        at=[at], 
        up=[up]
    )
    return R, T

def load_shape(shape, data_root):
    if shape is None:
        print('shape is none (ERROR)')
        return None
    device = torch.device("cuda:0")
    char_filename_node = shape.getElementsByTagName('string')[0]
    char_filename = char_filename_node.getAttribute('value')
    obj_filename = os.path.join(data_root, char_filename)
#     print(f'loading: {obj_filename}')
    mesh = load_objs_as_meshes([obj_filename], device=device)
#     print(f'loaded')
    verticies, faces = mesh.get_mesh_verts_faces(0)
    texture = mesh.textures.clone() if mesh.textures is not None else None
#     print(f'textures loaded')
    
    
    transform_node = shape.getElementsByTagName('transform')
    if len(transform_node) == 0:
        verticies[:, 2] *= -1
        return verticies, faces, texture
#     print(f'transform found')
    # apply transform
    transform_node = transform_node[0]
    model_matrix =  transform_node_to_matrix(transform_node)
    model_matrix = torch.from_numpy(model_matrix).cuda().double() 
#     print(f'apply transform')
    # make coordiantes homegenos
    new_row = torch.ones(1, verticies.shape[0], device=device)
    vetrices_homo = torch.cat((verticies.t(), new_row)).double() 
#     print(f'make coordiantes homegenos')
    # transform
    vetrices_world = torch.chain_matmul(model_matrix, vetrices_homo).t()[:, :3]
    return vetrices_world.float(), faces, texture

def mitsuba_scene_to_torch_3d_no_ground(master_scene, data_root):
    device = torch.device("cuda:0")
    master_doc = xml.dom.minidom.parse(master_scene)
    camera = master_doc.getElementsByTagName('sensor')[0]
    camera_transform = camera.getElementsByTagName('transform')[0]
    R, T = transform_node_to_R_T(camera_transform.getElementsByTagName('lookat')[0])

    cameras = OpenGLPerspectiveCameras(
        znear=0.1,
        zfar=10000,
        fov=15,
        degrees=True,
        device=device, 
        R=R, 
        T=T
    )
#     print('get camera')
    
    
    character = None
    tshirt = None
    
    shapes = master_doc.getElementsByTagName('shape')
    for i in range(len(shapes)):
        if shapes[i].getAttribute("id") == 'character':
            character = shapes[i]
        if shapes[i].getAttribute("id") == 'simulated':
            tshirt = shapes[i]

#     print('after for')
    tshirt_vetrices, tshirt_faces, tshirt_texture = load_shape(tshirt, data_root)
#     print('after tshirt')
    character_vetrices, character_faces, character_texture = load_shape(character, data_root)
#     print('after character_vetrices')
#     print('get shape')
    texTshirt = torch.ones_like(tshirt_vetrices).cuda()
    tex2 = torch.ones_like(character_vetrices).cuda()
    texTshirt[:, 1:] *= 0.0  # red
    # person
    tex2[:, 0] *= 0.88
    tex2[:, 1] *= 0.67
    tex2[:, 2] *= 0.41
    
    tex = torch.cat([texTshirt, tex2])[None]  # (1, 204, 3)
    textures = Textures(verts_rgb=tex.cuda())
#     print('get texture')
    verts = torch.cat([tshirt_vetrices, character_vetrices]).cuda()  #(204, 3)
    
    character_faces = character_faces + tshirt_vetrices.shape[0]  
#     print('get faces')
    faces = torch.cat([tshirt_faces, character_faces]).cuda()  
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
#     print('get mesh')
    optmization_input = {
        "texTshirt": texTshirt,
        "texOther": tex2,
        "verticies": verts, 
        "faces": faces,
        "R": R[0],
        "T": T[0]
    }
        
    return mesh, cameras, optmization_input


def get_body_image_from_mesh(cur_mesh, body_estimation, renderer, silhouette_renderer, orig_shape, bg_path='../data/backgrounds/indoor.jpg', cameras=None):
    images = add_background(bg_path, cur_mesh, renderer, silhouette_renderer, cameras)
        
    rendering_torch_input = (images[..., :3] - 0.5).permute((0, 3, 1, 2)).float()
    
    heatmap_avg, paf_avg = body_estimation.compute_heatmap_paf_avg(rendering_torch_input, orig_shape)
    candidate, subset = body_estimation.get_pose(heatmap_avg, paf_avg, orig_shape)
    rendering_torch_np =  images[0, ..., :3].detach().squeeze().cpu().numpy()
    canvas = copy.deepcopy(rendering_torch_np)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    
    return canvas, candidate, subset, heatmap_avg, paf_avg


def add_background(bg_path, mesh, renderer, silhouette_renderer, cameras=None):
    background = cv2.imread(bg_path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    background = torch.from_numpy(background / 255).unsqueeze(0).float().cuda()
    alpha = torch.ones((1, 512, 512, 1)).cuda()
    background = torch.cat([background, alpha], 3)
    
    if cameras is None:
        images = renderer(cur_mesh)
        silhouette = silhouette_renderer(mesh)   
    else:
        images = renderer(mesh, cameras=cameras)
        silhouette = silhouette_renderer(mesh, cameras=cameras)   
    

    alpha_mask = torch.cat([silhouette[..., 3], silhouette[..., 3], silhouette[..., 3], silhouette[..., 3]], 0)
    alpha_mask = alpha_mask.unsqueeze(0).permute((0, 2, 3, 1))

    final_images = where(alpha_mask > 0, images, background)
    return final_images

def add_background_batch(bg_paths, meshes, renderer, silhouette_renderer, cameras, lights):
    backgrounds = []
    batch_size = len(bg_paths)
    for bg_path in bg_paths:
        background = cv2.imread(bg_path)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        backgrounds.append(background)
        
    backgrounds = np.stack( backgrounds, axis=0 )
    backgrounds = torch.from_numpy(backgrounds / 255).float().cuda()
    
    alpha = torch.ones((batch_size, 512, 512, 1)).cuda()
    backgrounds = torch.cat([backgrounds, alpha], 3)
    
    
    images = renderer(meshes, cameras=cameras, lights=lights)
    silhouette = silhouette_renderer(meshes, cameras=cameras, lights=lights)
    alpha_masks = []
    for i in range(batch_size):
        alpha_mask = torch.stack([silhouette[i, ..., 3], silhouette[i, ..., 3], silhouette[i, ..., 3], silhouette[i, ..., 3]])
        
        alpha_mask = alpha_mask.unsqueeze(0).permute((0, 2, 3, 1))
        alpha_masks.append(alpha_mask)
    alpha_mask = torch.cat(alpha_masks).cuda()
    
    
    final_images = where(alpha_mask > 0, images, backgrounds)
    return final_images


def pose_loss_single_human(newHuman, oldHuman):
    if len(oldHuman) == 0:
        return -1
    if len(newHuman) == 0:
        return 0
    new_detected = 0
    old_detected = 0
    for part in range(18):
        if newHuman[0][part] != -1:
            new_detected += 1
        if oldHuman[0][part] != -1:
            old_detected += 1
    return new_detected / old_detected