import warnings
warnings.filterwarnings("ignore")
import os, shutil, logging, random
from easydict import EasyDict as edict
import numpy as np
import cv2 as cv
import time
import copy
import math
import datetime
from pathlib import Path
from tqdm import tqdm
import yaml

import torch
# import torchvision
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from torchsummary import summary

from argparse import ArgumentParser
from utilities import  dataset_ub4d, dataset_dnerf
# from utilities.stats import AverageMeter #, Stats
# from utils.vis_utils import get_visdom_env
from utilities.loss import RGB_Loss, Flow_Consistency_Loss, KPDistanceLoss,\
                        SinkhornSolver,ChamferDistanceLoss,KPNegDepthLoss
from utilities.plot_vis import plot_point_clouds,plot_test_img,plot_deform_codes,save_ply_pt_clds,\
                            save_depth_image, save_canonical_render, save_color_render#,plot_nerf_results
# from pytorch3d.loss import chamfer_distance

from utilities.tools import custom_collate_fn,Fixed_Seq_Batch_Sampler,\
                                    Stored_Random_Seq_Batch_Sampler,dump_yaml

# from run_test import test_Runner
from model import NPG
import pathlib
import sys
# sys.path
# sys.path.append('/BS/keytr_neus/work/')

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(sci_mode=False)
# torch.manual_seed(0)
# print('no of keypoints is 1000 and shape size=50 with positional encoding and grad true')
class Runner:
    def __init__(self,conf_path, mode, args):
        self.device = torch.device('cuda')

        self.writer = None # tensorboard

        # Path for the dataset and the experiment results
        self.conf = conf_path

        # This creates the path for the experiment data
        self.base_exp_dir = args.model_path
        os.makedirs(self.base_exp_dir, exist_ok=True)  
        
        # Reproducibility
        seed = int(self.conf['random_seed'])
        if seed >= 0:
            print('Seeding RNGs does not guarantee reproducibility!') # see https://pytorch.org/docs/stable/notes/randomness.html
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # torch.use_deterministic_algorithms(True) # requires setting an environment variable
            # Seed all possible sources of random numbers
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)


        # Load the dataset 
        self.batch_size = int(self.conf['batch_size'])

        dump_yaml(pathlib.Path(args.model_path),conf_path,cfg_file='data_config.yaml')

        self.dataset_type = str(self.conf['case_pose'])
        # if self.dataset_type == 'iphone':
            # self.train_dataset = dataset_iphone.myDataset(self.conf,mode)
        if self.dataset_type == 'ub4d':
            self.train_dataset = dataset_ub4d.myDataset(self.conf,mode)
        elif self.dataset_type == 'dnerf':
            self.train_dataset = dataset_dnerf.myDataset(self.conf,mode)
        # elif self.dataset_type == 'kfusion':
            # self.train_dataset = dataset_kfusion.myDataset(self.conf,mode)        
        else:
            assert False, "Could not recognize dataset type!"
        
        self.image_files  = self.train_dataset.dataset_ids


        # if self.pt_cld_grp:
        self.num_iter = int(self.conf['num_iter'])
        self.report_freq = int(self.conf['report_freq'])
        self.save_freq = int(self.conf['save_freq'])
        self.warm_up_end = int(self.conf['warm_up_end'])
        self.logging = int(self.conf['logging'])

        self.my_sampler = Stored_Random_Seq_Batch_Sampler(self.train_dataset,bs=self.batch_size)
            # self.my_sampler = Fixed_Seq_Batch_Sampler(self.train_dataset,bs=self.batch_size)
        
   
        # if self.pt_cld_grp:
        self.train_loader = data.DataLoader(self.train_dataset,\
                                        batch_sampler=self.my_sampler,shuffle=False,collate_fn=custom_collate_fn)#,\
                                        # num_workers=1,pin_memory=False,worker_init_fn=worker_init_fn)


        # # Training Parameters
        self.learning_rate = float(self.conf['learning_rate'])
        self.learning_rate_alpha = float(self.conf['learning_rate_alpha'])

        # Rasterizer
        self.radius = float(self.conf['radius'])
        self.pts_per_pixel = float(self.conf['pts_per_pixel']) 


        intermediate = 'pt_cld_grp'


        self.depth_path = os.path.join(self.base_exp_dir,'trial_data','depth')
        self.colour_render_path = os.path.join(self.base_exp_dir,'trial_data','colour_render')
        
        self.canonical_render_path = os.path.join(self.base_exp_dir,'trial_data','canonical_render')
        self.check_pt_path = os.path.join(self.base_exp_dir,'trial_data','check_point')
        # self.pt_cloud_path = os.path.join(self.base_exp_dir,'images',self.date_time_run,)
        self.plots_2d_path = os.path.join(self.base_exp_dir,'trial_data','plot_2d')
        self.optical_flow_path = os.path.join(self.base_exp_dir,'trial_data','flow_opt')
        self.deform_path = os.path.join(self.base_exp_dir,'trial_data','deform')
        self.save_pt_cld = os.path.join(self.base_exp_dir,'trial_data','pt_cld')

        Path(self.depth_path).mkdir(parents=True, exist_ok=True)
        Path(self.depth_path).mkdir(parents=True, exist_ok=True)
        Path(self.colour_render_path).mkdir(parents=True, exist_ok=True)
        Path(self.canonical_render_path).mkdir(parents=True, exist_ok=True)
        Path(self.plots_2d_path).mkdir(parents=True, exist_ok=True)
        Path(self.check_pt_path).mkdir(parents=True, exist_ok=True)
        Path(self.optical_flow_path).mkdir(parents=True, exist_ok=True)
        Path(self.deform_path).mkdir(parents=True, exist_ok=True)
        Path(self.save_pt_cld).mkdir(parents=True, exist_ok=True)


        self.factor_ = int(self.conf['factor'])
        h = int(self.conf['image_height']/self.factor_)
        w = int(self.conf['image_width']/self.factor_)
        self.image_size = tuple([h,w])

        self.total_frames = int(self.conf['num_frames'])
        # Writing in frames per second
        self.fps = int(self.conf['fps']) #frames per second in final video
        # Basis parameters
        self.n_keypoints = int(self.conf['key_points_no'])
        self.shape_basis_size = int(self.conf['shape_basis_size'])
        self.color_basis_size = int(self.conf['color_basis_size'])
        self.descriptor_dim = int(self.conf['phi_vector_dim'])


        # Loss initialization
        self.rgb_loss = RGB_Loss(self.conf,self.device,self.batch_size,self.image_size)
        self.flow_loss = Flow_Consistency_Loss(self.conf,self.device,save_path=self.optical_flow_path,bs = self.batch_size,img_size=self.image_size)
        self.chamfer_distance_pyt = ChamferDistanceLoss()
        # loss_chamfer is not a class. 
        self.assign_loss = SinkhornSolver(epsilon=0.01,iterations=1000)        
        self.kp_neg_loss = KPNegDepthLoss()
        self.kp_frame_dist = KPDistanceLoss(self.conf)

        # Loss weights

        self.w_asgn = float(self.conf['w_asgn'])
        self.w_flow = float(self.conf['w_flow'])
        self.w_cd = float(self.conf['w_cd'])
        self.w_rgb = float(self.conf['w_rgb'])
        self.w_d_plus = float(self.conf['w_d_plus'])
        self.w_d_kp = float(self.conf['w_d_kp'])

        

    def train(self):
        
        # initialize the model
        self.model = NPG(self.conf,self.device)

        # move model to gpu
        if torch.cuda.is_available():
            self.model.cuda()


        # Same for all frames. Defining the deformaton basis. b_dim = bat_size x basis_size X (3*n_keypoints)
        self.deformation_basis = torch.randn(size=(1,self.shape_basis_size, 3,self.n_keypoints),device=self.device)
        self.deformation_basis = torch.nn.functional.normalize(self.deformation_basis,dim=1)
        self.deformation_basis = torch.tensor(self.deformation_basis,requires_grad =True)       
        # For color basis
        # self.color_basis = torch.randn(size=(1,self.color_basis_size,self.n_keypoints),device=self.device)           
        # self.color_basis = torch.nn.functional.normalize(self.color_basis,dim=1)       
        # self.color_basis = torch.tensor(self.color_basis,requires_grad =False)
        # Feature Descriptor for canonical rendering/ FLow loss
        self.descriptors = torch.randn(size=(1,self.n_keypoints,self.descriptor_dim), device=self.device) #requires_grad =True
        self.descriptors = torch.nn.functional.normalize(self.descriptors,dim=2)


        # Optimizer
        self.params_to_train = []
        self.params_to_train+=[{'name':'model_network', 'params':self.model.parameters(), 'lr':self.learning_rate}]    
        self.params_to_train+=[{'name':'deformation_basis', 'params':self.deformation_basis, 'lr':self.learning_rate}]

        self.optimizer = torch.optim.Adam(self.params_to_train,lr=0.0)

        self.iter_step = 0
        # Tensorboard Writer
        self.writer = SummaryWriter(self.base_exp_dir)
   
        self.model.train()

        t_start = time.time()
        iter_sart = self.iter_step
        self.update_learning_rate()
        loader = iter(self.train_loader)
        torch.autograd.set_detect_anomaly(True)     
        for it in tqdm(range(iter_sart,self.num_iter,1)):
            
            
            data_samples = next(loader)

            # data_samples = next(training_loader_iter)
            imgs_,masks_,ref_fr_fg_pts_,flow_,frame_time_,proj_mat_,candidate_loc_,fg_points_, intrinsic_, pose_,_,ref_prj_mat_,fg_lengths_ = data_samples

            imgs,masks,ref_fr_fg_pts,flow, frame_time,proj_mat,candidate_loc,fg_points, intrinsic, pose, ref_prj_mat, fg_lengths = imgs_.to(self.device),masks_.to(self.device),\
                                            ref_fr_fg_pts_.to(self.device),flow_.to(self.device),frame_time_.to(self.device),\
                                                proj_mat_.to(self.device),candidate_loc_.to(self.device),fg_points_.to(self.device),\
                                                    intrinsic_.to(self.device), pose_.to(self.device), ref_prj_mat_.to(self.device),fg_lengths_.to(self.device)


            self.optimizer.zero_grad()
            
            outputs = self.model(frame_time=frame_time,proj_mat=proj_mat,deformation_basis=self.deformation_basis,feat_descriptors=self.descriptors,img_size=self.image_size)#,tuple(imgs.shape[2:4]))

            ## chamfer distance
            loss_cd = self.chamfer_distance_pyt(fg_points,outputs['proj_2d_keypts'],x_lengths=fg_lengths) # ref_fr_fg_pts,outputs['proj_2D_ref']
            loss_flow,canon_rendered,_ = self.flow_loss(imgs,outputs['pt_cld_canon'],flow,masks,intrinsic,pose,frame_time,[it,it])  # flow consistency
            ## loss_clr_py3d,py3d_rendered,_ = self.rgb_loss(imgs,outputs['pt_cld_color'],masks,intrinsic,pose) # rgb
            loss_kp_dist = self.kp_frame_dist(outputs['keypoint_3D'].permute(0,2,1),outputs['kp_dist_frame'].permute(0,2,1))
            neg_depth = self.kp_neg_loss(outputs['keypoint_3D'].permute(0,2,1),pose.unsqueeze(1))
            
            loss = loss_cd * self.w_cd \
                    + neg_depth * self.w_d_plus \
                    + loss_kp_dist * self.w_d_kp \
                    + loss_flow * self.w_flow 
                               
            
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                    
                #Logging into the Tensorboard
                if it%self.logging==0:

                    self.writer.add_scalar('Loss/Train loss', loss.item(), it)
                    self.writer.add_scalar('Loss/Chamfer Loss', loss_cd.item(), it)
                    self.writer.add_scalar('Loss/Depth Loss', neg_depth.item(), it)
                    self.writer.add_scalar('Loss/Flow Loss', loss_flow.item(), it) 
                    self.writer.add_scalar('Loss/KP Dist Loss', loss_kp_dist.item(), it)

                    self.writer.add_scalar('Learning Rate/L_Rate', self.optimizer.param_groups[0]['lr'], it)
                    self.writer.add_scalar('GPU/gpu_mem_total',torch.cuda.get_device_properties(0).total_memory/(1024*1024*1024),it)
                    self.writer.add_scalar('GPU/gpu_mem_reserved',torch.cuda.memory_reserved(0)/(1024*1024*1024),it)


                # Saving CheckPoints
                if it%self.save_freq==0:


                    checkpoint = {
                        'iterations': self.iter_step,
                        'model_network':self.model.state_dict(),
                        'deformation_basis':self.deformation_basis.detach().cpu(),
                        'descriptors':self.descriptors.detach().cpu(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_rate':self.optimizer.param_groups[0]['lr']
                    }

                    torch.save(checkpoint, os.path.join(self.check_pt_path, 'ckpt_cld_best_{:0>6d}.pth'.format(self.iter_step)))
                    print('Saved Checkpoint !')
                
                if it%self.report_freq==0:

                    temp = candidate_loc.cpu().numpy()[-1]
                    fty = self.image_files[int(temp)]
                    plot_test_img(outputs['proj_2d_keypts'][-1].cpu().numpy(),\
                        os.path.join(self.plots_2d_path, '{:0>8d}_{}_plot_2d.png'.format(it,fty)),\
                            self.image_size[1],self.image_size[0],masks[-1].cpu().numpy())

                    # save_canonical_render(canon_rendered[0].detach().cpu().numpy(),\
                    #     os.path.join(self.canonical_render_path, '{:0>8d}_{}_cn_render.png'.format(it,fty)))                    
                    
                    save_ply_pt_clds(outputs['keypoint_3D'].permute(0,2,1)[-1].cpu().numpy().squeeze(),\
                                                file_name=os.path.join(self.save_pt_cld, '{:0>8d}_{}_pointCloud.ply'.format(it,fty)),\
                                                rgb_points =(self.descriptors[...,:3][-1]*255).cpu().numpy().squeeze().astype(np.uint8))                   
                    
            self.update_learning_rate()

            self.iter_step += 1
            torch.cuda.empty_cache()
            # If completed number of iterations, then break
            if self.iter_step == self.num_iter:
                print('Training Completed')
                break
       
    # Updating Learning Rate
    def update_learning_rate(self):
        
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.num_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor            
if __name__ == '__main__':
    print('Experiment Started') # very important!

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    logging.getLogger('PIL').setLevel(logging.WARNING) # avoids excessive logging by PIL::PngImagePlugin.py

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    parser = ArgumentParser(description="Training script parameters")

    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('-c','--conf', type=str, required=True,help='path to yaml config file')
    parser.add_argument('-w','--model_path', type=str,  required=True, help='path to output path')
    parser.add_argument('-d','--data_dir', type=str,  required=True, help='path to source data path')
    parser.add_argument('-m','--mode', default='test', type=str,help='mode')
    parser.add_argument('-g','--gpu', default=0, type=int,help='index of gpu')
    args = parser.parse_args()

    # if args.is_continue=='True':
    #     dt_time = args.fold_id
    # else:
    #     current_time = datetime.datetime.now()
    #     dt_time = '{}-{}-{}_{}-{}-{}'.format(current_time.hour,current_time.minute,current_time.second,\
    #                 current_time.day,current_time.month,current_time.year)

    npg_confs = os.path.join(args.conf)

    with open(npg_confs, 'r') as f:
        npg_confs = yaml.safe_load(f)
    npg_confs['data_dir'] = args.data_dir

    torch.cuda.set_device(args.gpu)

    runner = Runner(npg_confs, args.mode, args)
    
    ## Train the model
    if args.mode == 'train':    
        runner.train()
    
