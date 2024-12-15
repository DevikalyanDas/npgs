#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json

import cv2 as cv
from glob import glob

from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    mask: np.array
    mask_path: str
    width: int
    height: int
    tid: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: list

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
#############################
def readOpenCVCameras(path_dir, transformsfile, white_background, extension=".png",flag=None):
    cam_infos = []

    rgb_image_lis = sorted(glob(os.path.join(path_dir, 'rgb/*.png')))
    mask_path_lis = sorted(glob(os.path.join(path_dir, 'mask/*.png')))
    n_images = len(rgb_image_lis)
    
    camera_dict = np.load(os.path.join(path_dir, transformsfile))

    for idx in range(n_images):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading {} camera {}/{}".format(transformsfile, idx+1, n_images))

        img_pth = rgb_image_lis[idx]
        mask_pth = mask_path_lis[idx]

        img = cv.imread(img_pth)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img_data = np.array(img, dtype=np.uint8)/255.0

        mask = cv.imread(mask_pth)
        mask = cv.cvtColor(mask,cv.COLOR_BGR2RGB)
        mask = np.array(mask, dtype=np.uint8)/255.0

        # im_ms_data = img_data*mask[:,:,None]
        #img_data *= mask 
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        arr = img_data[:,:,:3] #+ bg * (np.array([1,1,1])[None,...] - mask)

        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        mask =  Image.fromarray(np.array(mask*255.0, dtype=np.byte), "RGB")
        image_name = Path(img_pth).stem
        if flag=='test': 
            P1 = camera_dict['proj_mat_%d' % 27].astype(np.float32) # Camera Projection Matrixs
        else:
            P1 = camera_dict['proj_mat_%d' % idx].astype(np.float32)
        out = cv.decomposeProjectionMatrix(P1)
        K,R,t = out[0],out[1],out[2]
        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K
        intrinsics[:2,:3] =intrinsics[:2,:3]

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = (-R @ (t[:3] / t[3]))[:, 0]     


        R, t = pose[:3,:3], pose[:3,3:4].squeeze()

        py3d_2_colmap = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        # R = (py3d_2_colmap @ R).T
        # # # print(T.shape)
        # t = py3d_2_colmap @ t

        R = np.transpose(R)

        focalX = intrinsics[0,0]
        focalY = intrinsics[1,1]

        FovY = focal2fov(focalY,image.size[1])
        FovX = focal2fov(focalX,image.size[0])

        cam_infos.append(CameraInfo(uid=idx, R=R, T=t, FovY=FovY, FovX=FovX, image=image,mask=mask,
                        image_path=img_pth,mask_path=mask_pth, image_name=image_name, width=image.size[0], height=image.size[1],tid=idx))
        # if idx==1:

        # break
        
    sys.stdout.write('\n')
    return cam_infos

def fetchPlyCV(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    
    num_pts = len(vertices)
    colors = np.random.random((num_pts, 3)) / 255.0
    # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    normals = np.zeros((num_pts, 3))#np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    return positions,colors,normals
    # return BasicPointCloud(points=positions, colors=colors, normals=normals)

def readDynamicPCD(path, white_background, eval, extension=".png"):
    # same as colmap coordinate system: x-right, y-down and z-forward
    print("Reading Training Transforms")
    train_cam_infos = readOpenCVCameras(path, "projection_matrix.npz", white_background, extension,flag ='train')
    # print("Reading Novel Transforms") novel_projection_matrix_new1
    test_cam_infos = readOpenCVCameras(path, "projection_matrix.npz", white_background, extension, flag = 'test')
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []


    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path_list = []
    pcd_positions = []
    pcd_colors = []
    pcd_normals = []
    all_pcd_path = sorted(glob(os.path.join(path,"pt_cld",'*.ply')))
    for idx in range(len(train_cam_infos)):

        ply_path = all_pcd_path[idx]

        pos, col, nor = fetchPlyCV(ply_path)

        pcd_positions.append(pos)
        pcd_colors.append(col)
        pcd_normals.append(nor)

        ply_path_list.append(ply_path)

        
    pcd_positions_arr = np.stack(pcd_positions,axis=0)
    pcd_colors_arr = np.stack(pcd_colors,axis=0)
    pcd_normals_arr = np.stack(pcd_normals,axis=0)

    pcd_format = BasicPointCloud(points=pcd_positions_arr, colors=SH2RGB(pcd_colors_arr), normals=pcd_normals_arr)

    scene_info = SceneInfo(point_cloud=pcd_format,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path_list)
    return scene_info
####################################


####################################
def readDNeRFCameras(path_dir, transformsfile, white_background, extension=".png",flag=None,factor=1):
    cam_infos = []

    if flag=='train':

        # rgb_image_lis = sorted(glob(os.path.join(path_dir, 'rgb/*.png')))
        # mask_path_lis = sorted(glob(os.path.join(path_dir, 'mask/*.png')))
        data_json = 'transforms_train.json'
        img_mode = 'train'
        mask_mode = 'train_mask'
        frames_dict = os.path.join(path_dir, data_json)
        camera_dict = np.load(os.path.join(path_dir, transformsfile)) 
        with open(frames_dict) as json_file:
            contents1 = json.load(json_file)
            all_frames = contents1["frames"]
        # tot_num = len(all_frames)
    elif flag=='test':
        # rgb_image_lis = sorted(glob(os.path.join(path_dir, 'rgb/*.png')))
        # mask_path_lis = sorted(glob(os.path.join(path_dir, 'mask/*.png')))
        data_json = 'transforms_test.json'
        img_mode = 'test'
        mask_mode = 'test_mask'
        frames_dict = os.path.join(path_dir, data_json)
        camera_dict = np.load(os.path.join(path_dir, transformsfile))
        with open(frames_dict) as json_file:
            contents1 = json.load(json_file)
            all_frames = contents1["frames"]
        # tot_num = len(all_frames)
    elif flag=='val':

        # rgb_image_lis = sorted(glob(os.path.join(path_dir, 'rgb/*.png')))
        # mask_path_lis = sorted(glob(os.path.join(path_dir, 'mask/*.png')))
        data_json = 'mod_val_150.json'
        img_mode = 'train'
        mask_mode = 'train_mask'
        frames_dict = os.path.join(path_dir, data_json)
        camera_dict = np.load(os.path.join(path_dir, transformsfile)) 
        with open(frames_dict) as json_file:
            contents1 = json.load(json_file)
            all_frames = contents1["frames"]              
    tot_num = len(sorted(glob(os.path.join(path_dir, 'train/*.png'))))
    for num, frame in enumerate(all_frames):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading {} camera {}/{}".format(transformsfile, num+1, len(all_frames)))

        img_pth = os.path.join(path_dir, img_mode, Path(frame["file_path"]).stem + '.png')
        mask_pth = os.path.join(path_dir, mask_mode, Path(frame["file_path"]).stem + '.png')

        img = cv.imread(img_pth)
        mask = cv.imread(mask_pth)

        if data_json=='mod_val_150.json':
            img = np.ones((800,800,3))
            mask = np.ones((800,800,3))
        else:

            img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            mask = cv.cvtColor(mask,cv.COLOR_BGR2RGB)

        if factor != 1:
            # resize the images
            shp_ = img.shape[:2]
            shp_img = tuple(reversed(tuple(int(ti/factor) for ti in shp_)))
            img = cv.resize(img,shp_img,interpolation= cv.INTER_LINEAR)
            mask = cv.resize(mask,shp_img,interpolation= cv.INTER_LINEAR)
            mask = np.where(mask<128,0,255)

        img_data = np.array(img, dtype=np.uint8)/255.0
        mask = np.array(mask, dtype=np.uint8)/255.0

        # im_ms_data = img_data*mask[:,:,None]
        img_data *= mask 
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        arr = img_data[:,:,:3] #+ bg * (np.array([1,1,1])[None,...] - mask)

        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        mask =  Image.fromarray(np.array(mask*255.0, dtype=np.byte), "RGB")
        image_name = Path(img_pth).stem

        t_idx = np.ceil(frame['time']*(tot_num-1))

        # if t_idx == tot_num:
        #     t_idx -= 1
        # if flag=='test': 
        #     P1 = camera_dict['proj_mat_%d' % 55].astype(np.float32) # Camera Projection Matrixs
        # else:
        # if flag=='train':
        #     num=1
        P1 = camera_dict['proj_mat_%d' % num].astype(np.float32)
        out = cv.decomposeProjectionMatrix(P1)
        K,R,t = out[0],out[1],out[2]
        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K
        intrinsics[:2,:3] =intrinsics[:2,:3]

        if factor != 1:
            intrinsics[0,0] = intrinsics[0,0]/factor
            intrinsics[1,1] = intrinsics[1,1]/factor
            intrinsics[0,2] = intrinsics[0,2]/factor
            intrinsics[1,2] = intrinsics[1,2]/factor

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = (-R @ (t[:3] / t[3]))[:, 0]     


        R, t = pose[:3,:3], pose[:3,3:4].squeeze()

        # py3d_2_colmap = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        # R = (py3d_2_colmap @ R).T
        # # # print(T.shape)
        # t = py3d_2_colmap @ t

        R = np.transpose(R)

        focalX = intrinsics[0,0]
        focalY = intrinsics[1,1]

        FovY = focal2fov(focalY,image.size[1])
        FovX = focal2fov(focalX,image.size[0])

        cam_infos.append(CameraInfo(uid=num, R=R, T=t, FovY=FovY, FovX=FovX, image=image,mask=mask,
                        image_path=img_pth,mask_path=mask_pth, image_name=image_name, width=image.size[0], height=image.size[1],tid=int(t_idx)))
        # if num==1:

        #     break
        
    sys.stdout.write('\n')
    return cam_infos


def readDnerfData(path, white_background, eval, extension=".png"):
    # same as colmap coordinate system: x-right, y-down and z-forward
    print("Reading Training Transforms")
    train_cam_infos = readDNeRFCameras(path, "projection_matrix.npz", white_background, extension,flag ='train',factor=2)
    # print("Reading Novel Transforms") novel_projection_matrix_new1
    test_cam_infos = readDNeRFCameras(path, "novel_projection_matrix.npz", white_background, extension, flag = 'test',factor=2)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []


    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path_list = []
    pcd_positions = []
    pcd_colors = []
    pcd_normals = []
    all_pcd_path = sorted(glob(os.path.join(path,"pt_cld_40",'*.ply')))
    for idx in range(len(train_cam_infos)):

        ply_path = all_pcd_path[idx]

        pos, col, nor = fetchPlyCV(ply_path)

        pcd_positions.append(pos)
        pcd_colors.append(col)
        pcd_normals.append(nor)

        ply_path_list.append(ply_path)
        # if idx==1:
        #     break
        
    pcd_positions_arr = np.stack(pcd_positions,axis=0)
    pcd_colors_arr = np.stack(pcd_colors,axis=0)
    pcd_normals_arr = np.stack(pcd_normals,axis=0)

    pcd_format = BasicPointCloud(points=pcd_positions_arr, colors=SH2RGB(pcd_colors_arr), normals=pcd_normals_arr)

    scene_info = SceneInfo(point_cloud=pcd_format,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path_list)
    return scene_info
####################################

####################################
def read_json_iphone_cameras(data,path_dir):
    # scene_center = np.array([-0.20061972737312317, 0.1705830693244934, -1.1717479228973389 ])
    # scene_scale = 0.2715916376300303
    with open(os.path.join(path_dir,'scene.json')) as fs:
        scene_json = json.load(fs)
    with open(os.path.join(path_dir,'extra.json')) as ex:
        extra_json = json.load(ex)

    scene_center = scene_json['center']
    scene_scale = float(scene_json['scale'])
    dataset_factor = float(extra_json['factor'])

    with open(data) as f:
        cam_dict = json.load(f)
        orientation=np.asarray(cam_dict["orientation"])
        position=np.asarray(cam_dict["position"])
        focal_length=cam_dict["focal_length"]
        principal_point=np.asarray(cam_dict["principal_point"])
        image_size=np.asarray(cam_dict["image_size"])
        skew=cam_dict["skew"]
        pixel_aspect_ratio=cam_dict["pixel_aspect_ratio"]
        radial_distortion=np.asarray(cam_dict["radial_distortion"])
        tangential_distortion=np.asarray(cam_dict["tangential_distortion"])

        position -= np.array(scene_center)
        position *= scene_scale

        translation = - orientation @ position
        scale_factor_x = focal_length/dataset_factor
        scale_factor_y = focal_length /dataset_factor
        principal_point_x = principal_point[0]/dataset_factor
        principal_point_y = principal_point[1]/dataset_factor

        image_size_y = image_size[1]
        image_size_x = image_size[0]

        R = orientation
        T = translation

        py3d_2_colmap = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        R = np.transpose(R)
        # # # print(T.shape)


        f_x = scale_factor_x
        f_y = scale_factor_y
  

    # intrinsics = np.eye(4)
    # intrinsics[:3,:3] = intrinsic


    return R, T, f_x, f_y

def readIphoneCameras(path_dir, transformsfile, white_background, extension=".png"):
    cam_infos = []
    if transformsfile == 'train':

        rgb_image_lis = sorted(glob(os.path.join(path_dir, 'train_image','*.png')))
        mask_path_lis = sorted(glob(os.path.join(path_dir, 'train_mask','*.png')))
        camera_dict = sorted(glob(os.path.join(path_dir,'train_camera', '*.json')))
    if transformsfile == 'test':
        if os.path.exists(os.path.join(path_dir, "test_image")):
            rgb_image_lis = sorted(glob(os.path.join(path_dir, 'test_image','*.png')))
            mask_path_lis = sorted(glob(os.path.join(path_dir, 'test_mask','*.png')))
            camera_dict = sorted(glob(os.path.join(path_dir,'test_camera', '*.json')))
            novel_pose_flag = True
        else:
            rgb_image_lis = sorted(glob(os.path.join(path_dir, 'train_image','*.png')))
            mask_path_lis = sorted(glob(os.path.join(path_dir, 'train_mask','*.png')))
            camera_dict = sorted(glob(os.path.join(path_dir,'train_camera', '*.json')))
            novel_pose_flag = False
            print("Found no gt test pose. Fixing a train pose")            
    n_images = len(rgb_image_lis)
    
    
    for idx in range(n_images):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading {} camera {}/{}".format(transformsfile, idx+1, n_images))
        sys.stdout.flush()        
        img_pth = rgb_image_lis[idx]
        mask_pth = mask_path_lis[idx]

        img = cv.imread(img_pth)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img_data = np.array(img, dtype=np.uint8)/255.0

        mask = cv.imread(mask_pth)
        mask = cv.cvtColor(mask,cv.COLOR_BGR2RGB)
        mask = np.array(mask, dtype=np.uint8)/255.0

        # im_ms_data = img_data*mask[:,:,None]

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        arr = img_data[:,:,:3] #+ bg * (1 - mask)

        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        mask =  Image.fromarray(np.array(mask*255.0, dtype=np.byte), "RGB")

        image_name = Path(img_pth).stem

        cam_file= camera_dict[idx]
        if transformsfile == 'test':
            if novel_pose_flag:
                cam_file= camera_dict[idx]
            else:
                cam_file= camera_dict[0]

        R, t, f_x, f_y = read_json_iphone_cameras(cam_file,path_dir)
            
        # R = np.transpose(R)

        focalX = f_x
        focalY = f_y

        FovY = focal2fov(focalY,image.size[1])
        FovX = focal2fov(focalX,image.size[0])

        t_idx = int(Path(img_pth).stem.split("_")[-1])

        # sys.stdout.write("Image size {}x{}".format(image.size[0],image.size[1]))
        cam_infos.append(CameraInfo(uid=idx, R=R, T=t, FovY=FovY, FovX=FovX, image=image,mask=mask,
                        image_path=img_pth,mask_path=mask_pth, image_name=image_name, width=image.size[0], height=image.size[1],tid=t_idx))
    
    sys.stdout.write('\n')
    return cam_infos
def readNerfiesdata(path, white_background, eval, extension=".png"):
    # same as colmap coordinate system: x-right, y-down and z-forward
    
    train_cam_infos = readIphoneCameras(path, "train", white_background, extension)
    test_cam_infos = readIphoneCameras(path, "test", white_background, extension)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []


    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path_list = []
    pcd_positions = []
    pcd_colors = []
    pcd_normals = []
    all_pcd_path = glob(os.path.join(path,"pt_cld",'*.ply'))
    # with open(os.path.join(path,'scene.json')) as fs:
    #     scene_json = json.load(fs)
    for idx in range(len(train_cam_infos)):

        ply_path = all_pcd_path[idx]

        pos, col, nor = fetchPlyCV(ply_path)

        pcd_positions.append(pos)
        pcd_colors.append(col)
        pcd_normals.append(nor)

        ply_path_list.append(ply_path)
    
        
    pcd_positions_arr = np.stack(pcd_positions,axis=0)
    pcd_colors_arr = np.stack(pcd_colors,axis=0)
    pcd_normals_arr = np.stack(pcd_normals,axis=0)

    pcd_format = BasicPointCloud(points=pcd_positions_arr, colors=SH2RGB(pcd_colors_arr), normals=pcd_normals_arr)

    scene_info = SceneInfo(point_cloud=pcd_format,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path_list)
    return scene_info
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DynamicPCD" : readDynamicPCD,
    "Nerfies" : readNerfiesdata,
    "Dnerf" : readDnerfData,
}