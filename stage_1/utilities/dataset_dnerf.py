import os
import torch
# vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import random
import math
import os
from glob import glob
import json

import logging
from torch.utils.data import Dataset
from utilities.tools import fps
import pytorch3d
from pytorch3d.ops import sample_farthest_points as sfp

from utilities.tools import PLYWriter
from pathlib import Path
# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R#.transpose()
    pose[:3, 3] = (-R @ (t[:3] / t[3]))[:, 0]#(t[:3] / t[3])[:, 0]

    return intrinsics, pose


class myDataset(Dataset):
    '''
        img, mask, frame_time, pose data loader
    '''
    def __init__(self, conf,mode):
        super(myDataset, self).__init__()
        logging.info('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.dtype = torch.get_default_dtype()


        self.data_dir = conf['data_dir']
  
        self.render_cameras_name = conf['render_cameras_name']

        self.factor = int(conf['factor'])
        
        self.reference_frame = int(conf['ref_frame'])

        self.mode = mode

        self.novel_pose = bool(conf['novel_pose'])
        self.novel_pose_available = bool(conf['novel_pose_available'])
        self.novel_pose_idx = int(conf['novel_pose_idx'])
        self.novel_time = int(conf['novel_time'])
        self.novel_time_idx = float(conf['novel_time_idx'])

        if self.mode == 'train':
            self.img_mode = 'train'
            self.mask_mode = 'train_mask'
        elif self.mode == 'test':
            if self.novel_pose:
                if self.novel_pose_available:
                    self.img_mode = 'test'
                    self.mask_mode = 'test_mask'
                else:
                    self.img_mode = 'train'
                    self.mask_mode = 'train_mask'                    
            else:
                self.img_mode = 'train'
                self.mask_mode = 'train_mask'

        camera_dict = os.path.join(self.data_dir,'{}.json'.format(self.render_cameras_name))

        self.camera_dict = camera_dict
        with open(self.camera_dict) as json_file:
            contents = json.load(json_file)
            self.fov = contents['camera_angle_x']
            self.all_frames = contents["frames"]

        temp_cam_dict1 = np.load(os.path.join(self.data_dir, 'projection_matrix.npz'))
        temp_cam_dict = [temp_cam_dict1['proj_mat_%d' % idx].astype(np.float32) for idx in range(len(temp_cam_dict1))]
        self.temp_cam_dict = temp_cam_dict

        # Get the names of the images in the folder
        # self.images_lis = sorted(glob(os.path.join(self.data_dir, 'train/*.png')))
        # self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'train_mask/*.png')))
        self.flow_lis = sorted(glob(os.path.join(self.data_dir, 'flow/flow/*.pt')))
        self.flow_flag = 'pt'
        if len(self.flow_lis) == 0:
            self.flow_lis = sorted(glob(os.path.join(self.data_dir, 'flow/flow/*.npy')))
            self.flow_flag = 'npy'
        
        self.total_frames = len(self.flow_lis)

        if self.novel_pose:
            if self.novel_pose_available:
                self.novel_cameras_name = str(conf['novel_cameras_name'])
                self.camera_dict = os.path.join(self.data_dir, '{}.json'.format(self.novel_cameras_name))
                temp_cam_dict1 = np.load(os.path.join(self.data_dir, 'novel_projection_matrix.npz'))
                temp_cam_dict = [temp_cam_dict1['proj_mat_%d' % idx].astype(np.float32) for idx in range(len(temp_cam_dict1))]
                self.temp_cam_dict = temp_cam_dict                
                with open(self.camera_dict) as json_file:
                    contents1 = json.load(json_file)
                    self.fov = contents1['camera_angle_x']
                    self.all_frames = contents1["frames"]
            else:
                # Fixing a camera pose
                # self.all_frames = [self.all_frames[self.novel_pose_idx] for idx in range(len(self.all_frames))]
                temp_cam_dict1 = np.load(os.path.join(self.data_dir, 'projection_matrix.npz'))
                temp_cam_dict = [temp_cam_dict1['proj_mat_%d' % self.novel_pose_idx].astype(np.float32) for idx in range(len(temp_cam_dict1))]
                self.temp_cam_dict = temp_cam_dict
        if self.mode=='train':
            self.dataset_ids= [Path(i["file_path"]).stem for i in self.all_frames]
        elif self.mode=='test':
            if self.novel_pose:
                if self.novel_pose_available:
                    self.dataset_ids= [Path(i["file_path"]).stem for i in self.all_frames]
                else:
                    self.dataset_ids= [Path(i["file_path"]).stem for i in self.all_frames] # Fixed camera pose but different ids
            else:
                self.dataset_ids= [Path(i["file_path"]).stem for i in self.all_frames]

        self.img_all = []
        self.mask_all = []
        self.depth_all = []
        self.flow_all = []
        self.time_all = []
        self.proj_mat_all = []
        self.candidate_loc_all = []
        self.fg_points_all = [] 
        self.intrinsics_all = [] 
        self.pose_all = []
        self.frame_time_nov_all = []
        self.fg_lengths_all = []
        self.name_index = []
        self.load_data_all()
        logging.info('Load data: End')

    def load_data_all(self):

        for num,index in enumerate(self.all_frames):
            
            frame = index
            img_pth = os.path.join(self.data_dir, self.img_mode, Path(frame["file_path"]).stem + '.png')
            mask_pth = os.path.join(self.data_dir, self.mask_mode, Path(frame["file_path"]).stem + '.png')
            
            # mask_pth = self.masks_lis[num]
            flow_pth = self.flow_lis[num]

            img = cv.imread(img_pth)
            img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.uint8)

            shp_ = img.shape[:2]
            shp_img = tuple(reversed(tuple(int(ti/self.factor) for ti in shp_)))

            if self.factor != 1:    
                img = cv.resize(img,shp_img,interpolation= cv.INTER_LINEAR)
                
            img = torch.from_numpy(img)
            img = img.permute(2,0,1).float()
            # Mask
            mask = cv.imread(mask_pth,cv.IMREAD_GRAYSCALE)
            
            if self.factor != 1:    
                mask = cv.resize(mask,shp_img,interpolation= cv.INTER_LINEAR)
            index_mask = np.argwhere(mask>0)
            # print(index_mask.shape)
            # index_mask = index_mask/np.array([mask.shape[0],mask.shape[1]])
            mask = np.array(mask, dtype=np.uint8)
            mask = np.where(mask==0,0.,1.)
            mask = torch.from_numpy(mask).type(torch.float)

            mask = mask[None,:]

            # Flow: can be pt format of npy format
            if self.flow_flag == 'pt':
                flow = torch.load(flow_pth).permute(2,0,1) #torch.zeros(1)#
            else:
                flow = torch.from_numpy(np.load(flow_pth)).permute(2,0,1).float()

            depth = torch.tensor([2.0])
            # selection of random foreground points for optimal transport 
            sel_index = random.choices(index_mask,k=int(self.conf['num_candidate']))
            # sel_index = fps(index_mask,self.conf.get_int('num_candidate'))

            # #### using farthest point sampling from pytorch 3d ###
            # sel_index,_ = sfp(torch.tensor(index_mask)[None,...],K=self.conf.get_int('num_candidate'))
            # sel_index = sel_index.squeeze()
            candidate_loc = torch.from_numpy(np.array([num],dtype=np.float32).squeeze()) # for assignment loss
            
            fg_points = torch.tensor([tuple(reversed(i)) for i in index_mask]) # for chamfer distance
            fg_lengths = torch.tensor(fg_points.shape[0])

            # Selection of frame time for shape coefficient
            time_index = np.ceil(frame['time']*(self.total_frames-1))

            frame_time = np.array([time_index],dtype=np.float32).squeeze()#np.array((self.images_lis[index].split('/')[-1]).split('.')[0],dtype=np.float32)
            
            frame_time = torch.from_numpy(frame_time)

            # flow = torch.tensor([2.0])#torch.from_numpy(np.load(feat_path).squeeze())

            # For Novel pose time view synthesis
            proj_mat, intrinsics, pose = self.dnerf_camera_matrix(num,img.permute(1,2,0))

            if self.novel_time:    
                frame_time_nov = torch.from_numpy(np.array(self.all_frames[self.novel_time_idx]['time'],dtype=np.float32))
            else:
                frame_time_nov = torch.tensor([2.0])

            self.img_all.append(img)
            self.mask_all.append(mask)
            self.time_all.append(frame_time)
            self.depth_all.append(depth)
            self.flow_all.append(flow)
            self.proj_mat_all.append(torch.from_numpy(proj_mat))
            self.candidate_loc_all.append(candidate_loc)
            self.name_index.append(index)
            self.fg_points_all.append(fg_points)
            self.intrinsics_all.append(torch.from_numpy(intrinsics))
            self.pose_all.append(torch.from_numpy(pose))
            self.frame_time_nov_all.append(frame_time_nov)
            self.fg_lengths_all.append(fg_lengths)

        logging.info('Found {} {} cameras'.format(len(self.intrinsics_all),self.mode))

    def __len__(self):
        return len(self.dataset_ids)

    def __getitem__(self,index):

        img = self.img_all[index]
        mask = self.mask_all[index]
        # depth = self.depth_all[index]
        flow = self.flow_all[index]
        frame_time = self.time_all[index]
        proj_mat = self.proj_mat_all[index]
        candidate_loc = self.candidate_loc_all[index]
        fg_points = self.fg_points_all[index]
        intrinsics = self.intrinsics_all[index]
        pose = self.pose_all[index]
        frame_time_nov = self.frame_time_nov_all[index]

        ref_fr_fg = self.fg_points_all[self.reference_frame]
        ref_prj_mat = self.proj_mat_all[self.reference_frame]

        fg_leghts_num = self.fg_lengths_all[index]

        return img, mask, ref_fr_fg, flow, frame_time, proj_mat, candidate_loc,fg_points, intrinsics, pose,frame_time_nov,ref_prj_mat,fg_leghts_num 

    def dnerf_camera_matrix(self,idx,img1):

        # # np.printoptions(suppress=True)
        # if self.warm_up:
        #     idx = idx + self.cam_idx_start #this is for temporary. test case of 130-149 for cactus


        P1 = self.temp_cam_dict[idx]
        out = cv.decomposeProjectionMatrix(P1)
        K,R,t = out[0],out[1],out[2]
        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K
        intrinsics[:2,:3] =intrinsics[:2,:3]

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = (-R @ (t[:3] / t[3]))[:, 0]#/self.factor #(-R @ (t[:3] / t[3]))[:, 0]#(t[:3] / t[3])[:, 0]

        prj_m = intrinsics@pose

        return np.float32(prj_m[:3,:4]),np.float32(intrinsics), np.float32(pose)       
        


        