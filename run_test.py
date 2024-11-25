import warnings
warnings.filterwarnings("ignore")
import os, shutil, logging, random,glob,json
import imageio
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import time
import copy
import datetime
from pathlib import Path
from tqdm import tqdm
import yaml
# import trimesh
import torch
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
# from torchsummary import summary
# from utilities.args import ParseArgs
from argparse import ArgumentParser
from utilities import dataset_ub4d, dataset_dnerf

# from utils.vis_utils import get_visdom_env

from utilities.plot_vis import plot_point_clouds,plot_test_img,save_ply_pt_clds,save_depth_image, save_canonical_render, save_color_render,plot_overlay_pts_img

from utilities.tools import Random_Seq_Batch_Sampler,custom_collate_fn
from pytorch3d.renderer import look_at_view_transform, BlendParams, \
                                PointsRasterizationSettings, PointsRenderer, PulsarPointsRenderer,\
                                PointsRasterizer, AlphaCompositor,\
                                NormWeightedCompositor
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer.compositing import alpha_composite
# from pt_nerf import PointNerfRenderer
from model import NPG
import sys
# sys.path
# sys.path.append('/BS/keytr_neus/work/')

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)

class test_Runner:
    def __init__(self,conf_path, mode, args):
        self.device = torch.device('cuda')
        self.writer = None # tensorboard
        
        self.conf = conf_path
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

        self.batch_size = int(self.conf['batch_size'])
        self.my_sampler = Random_Seq_Batch_Sampler(self.train_dataset,bs=self.conf['batch_size'])
        # self.train_loader = data.DataLoader(self.train_dataset,\
        #                                 batch_sampler=self.my_sampler,shuffle=False,collate_fn=custom_collate_fn)
        self.train_loader = data.DataLoader(self.train_dataset,\
                                        batch_size=self.batch_size,shuffle=False,collate_fn=custom_collate_fn)
        
        self.factor_ = int(self.conf['factor'])
        h = int(self.conf['image_height']/self.factor_)
        w = int(self.conf['image_width']/self.factor_)
        self.image_size = tuple([h,w])
        self.novel_time = self.conf['novel_time']
        
        # catergory of weights
        self.pt_cld_grp = bool(self.conf['pt_cld_grp'])

        # Parameters for the rasterizer 
        self.radius = float(self.conf['radius'])
        self.pts_per_pixel = int(self.conf['pts_per_pixel'])        

        self.num_frames = int(self.conf['num_frames'])
  
        # save outputs in terms of pointclouds and depth

        self.base_exp_dir = args.model_path #path where all the weights are stored
        self.name_files = self.train_dataset.dataset_ids
        
        self.num_cp = args.test_iteration
        self.check_pt_path = os.path.join(self.base_exp_dir,'checkpoint','ckpt_cld_best_{:0>6d}.pth'.format(self.num_cp))
        
        self.test_video_path = os.path.join(self.base_exp_dir,mode,'test_{}'.format(self.num_cp),'video')
        self.test_colour_path = os.path.join(self.base_exp_dir,mode,'test_{}'.format(self.num_cp),'color')
        self.test_py3d_colour_path = os.path.join(self.base_exp_dir,mode,'test_{}'.format(self.num_cp),'py3d color')
        self.test_overlay_path = os.path.join(self.base_exp_dir,mode,'test_{}'.format(self.num_cp),'overlay')
        self.test_canonical_path = os.path.join(self.base_exp_dir,mode,'test_{}'.format(self.num_cp),'canonical')
        self.test_pt_cld_path = os.path.join(self.base_exp_dir,mode,'test_{}'.format(self.num_cp),'pt_cld')
        self.test_2d_proj_path = os.path.join(self.base_exp_dir,mode,'test_{}'.format(self.num_cp),'proj_2d')
        self.test_basis_pt_cld_path = os.path.join(self.base_exp_dir,mode,'test_{}'.format(self.num_cp),'basis_pt_cld')

        print(self.base_exp_dir)

        Path(self.test_video_path).mkdir(parents=True, exist_ok=True)
        Path(self.test_colour_path).mkdir(parents=True, exist_ok=True)
        Path(self.test_py3d_colour_path).mkdir(parents=True, exist_ok=True)
        Path(self.test_canonical_path).mkdir(parents=True, exist_ok=True)
        Path(self.test_overlay_path).mkdir(parents=True, exist_ok=True)
        Path(self.test_pt_cld_path).mkdir(parents=True, exist_ok=True)
        Path(self.test_2d_proj_path).mkdir(parents=True, exist_ok=True)
        Path(self.test_basis_pt_cld_path).mkdir(parents=True, exist_ok=True)
        self.coeff_def_dict = {}
        # Basis parameters
        self.n_keypoints = int(self.conf['key_points_no'])
        self.shape_basis_size = int(self.conf['shape_basis_size'])
        self.color_basis_size = int(self.conf['color_basis_size'])
        self.descriptor_dim = int(self.conf['phi_vector_dim'])

    def test(self):

        # initialize the model
        self.model = NPG(self.conf,self.device)

        # move model to gpu
        if torch.cuda.is_available():
            self.model.cuda()


        self.model.eval()
        
        self.checkpoint = torch.load(self.check_pt_path)

        self.model.load_state_dict(self.checkpoint['model_network'])

        self.descriptors = torch.tensor(self.checkpoint['descriptors']).to(self.device)
        self.deformation_basis = torch.tensor(self.checkpoint['deformation_basis']).to(self.device)
        self.individual_features  = torch.tensor(self.checkpoint['color_basis']).to(self.device)



        with torch.no_grad():

            t_start = time.time()

            self.max_v = []
            self.min_v = []
            for it, data_samples in tqdm(enumerate(self.train_loader)):
                imgs_,masks_,ref_fg_pts_,flow_,frame_time_,proj_mat_,candidate_loc_,fg_points_, intrinsic_, pose_, frame_time_nov_,ref_prj_mat_,_ = data_samples

                imgs,masks,ref_fg_pts,flow, frame_time,proj_mat,candidate_loc,fg_points, intrinsic, pose, frame_time_nov,ref_prj_mat = imgs_.to(self.device),masks_.to(self.device),\
                                                ref_fg_pts_.to(self.device),flow_.to(self.device),frame_time_.to(self.device),\
                                                    proj_mat_.to(self.device),candidate_loc_.to(self.device),fg_points_.to(self.device),\
                                                        intrinsic_.to(self.device), pose_.to(self.device),frame_time_nov_.to(self.device),ref_prj_mat_.to(self.device)


                bs,ch,h,w = imgs.shape
                
                if self.novel_time:
                    ft_seq = frame_time_nov
                else:
                    ft_seq = frame_time
                outputs = self.model(frame_time=ft_seq,proj_mat=proj_mat,deformation_basis=self.deformation_basis,\
                            feat_descriptors=self.descriptors[:1].repeat(bs,1,1),img_size=self.image_size)
                    

                # canon_rendered,depth_fin_keytr = self.softRASTERIZER(outputs['pt_cld_canon'],intrinsic,pose,self.image_size,bs)
                # py3d_color_rendered,depth_fin1 = self.softRASTERIZER(outputs['pt_cld_color'],intrinsic,pose,self.image_size)
                

                # if self.pt_cld_grp:
                    # updated_pt_nerf_features = self.ptnf_features.expand(bs,self.n_keypoints,self.pt_nerf_feat_dim)
                    # gathered_rot = None                     
     
                self.save_pt_clds(outputs['keypoint_3D'].permute(0,2,1),candidate_loc,self.descriptors[...,:3])
                # if self.novel_pose:
                    # masks1 = self.fixed_mask
                # else:
                    # masks1 = masks
                # self.save_test_image(depth_fin,candidate_loc,flag='depth') #*masks1.squeeze()
                # self.save_2d_projection(outputs['proj_2d_keypts'],candidate_loc,masks1)#masks ,self.fixed_mask
                # self.save_overlay_pts_img(outputs['proj_2d_keypts'],candidate_loc,imgs.permute(0,2,3,1))
                self.save_deformation_coeff(outputs['deform_coeff'].squeeze(),candidate_loc)
                # break
            # # break
            # self.save_test_video(self.test_canonical_path,flag='canonical')
            # self.save_test_video(self.test_py3d_colour_path,flag='color')
            # self.save_test_video(self.test_colour_path,flag='color')
            # self.save_test_video(self.test_depth_path,flag='depth')
            # self.save_test_video(self.conf['dataset.data_dir'],flag='ori')
            # self.save_test_video(self.conf['dataset.data_dir'],flag='ori_depth') 
            # self.save_test_video(self.test_2d_proj_path,flag='proj_2d')
            # self.save_test_video(self.test_overlay_path,flag='overlay')

            # self.save_gif_video(self.test_colour_path,flag='color')
            # self.save_gif_video(self.test_depth_path,flag='depth')
            # self.save_gif_video(self.test_2d_proj_path,flag='proj_2d')

            

            # Save dictionary of deform coeff into json file:
            json_name = os.path.join(self.test_video_path,"deform_coeff.json")
            json.dump( self.coeff_def_dict, open(json_name, 'w' ) )
            
            t_end = time.time() - t_start
            print('Inferencing complete! Time Taken ::: {}'.format(t_end))
            # print('Max values:{}'.format(max(self.max_v)))
            # print('Min values:{}'.format(min(self.min_v)))

    def unflatten_pred(self,pred):
        res = pred.transpose(-1, -2)
        m = res.shape[-1]

        # side = round(m ** (0.5))
        return res.reshape(*res.shape[:-1], self.nerf_resolution[0], self.nerf_resolution[1])
    
    def softRASTERIZER(self,pt_cld,intrinsic,pose,img_size,bs):

        R, T = pose[:,:3,:3], pose[:,:3,3:4].squeeze()

        intrinsics= intrinsic[:,:3,:3]

        # point_cloud = Pointclouds(points=pt_cloud.squeeze(), features=feat)
        img_size_tensor = torch.from_numpy(np.array(img_size))

        img_size_t = torch.stack([img_size_tensor for _ in range(bs)])        
        cameras = cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=intrinsics,\
                                image_size=img_size_t)
        
        # blend_params = BlendParams(sigma=1e-5, gamma=1e-4)
        raster_settings = PointsRasterizationSettings(
                    image_size=img_size, 
                    radius = self.radius, #np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                    points_per_pixel = self.pts_per_pixel
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
                    rasterizer=rasterizer,
                    compositor=AlphaCompositor())
            # background_color=(0,1,0)
        images = renderer(pt_cld)
        pt_fragments = rasterizer(pt_cld)

        # r = raster_settings.radius
        # dists2 = pt_fragments.dists
        # weights = 1.0-dists2/(r*r + 1e-6)
        # pt_frag = weights * 
        return images,pt_fragments.zbuf

    def save_deformation_coeff(self,def_coeff,frame_time):
        frame_time = frame_time.detach().cpu().numpy().squeeze()
        def_coeff = def_coeff.detach().cpu().numpy().squeeze()
        for num,ft in enumerate(frame_time):
            name = self.name_files[int(ft)]
            self.coeff_def_dict['frame_{}'.format(name)] = list(def_coeff[num].astype(float))

    
    def save_overlay_pts_img(self,pt_2d,frame_time,img_d):
        frame_time = frame_time.detach().cpu().numpy()
        for num,ft in enumerate(frame_time):
            name = self.name_files[int(ft)]
            plot_overlay_pts_img(pt_2d[num].detach().cpu().numpy().squeeze(),\
                os.path.join(self.test_overlay_path, '{}_overlay.png'.format(name)),\
                self.image_size[1],self.image_size[0],img_d[num].detach().cpu().numpy().squeeze())
    def save_2d_projection(self,pt_2d,frame_time,mask_d):
        frame_time = frame_time.detach().cpu().numpy()

        for num,ft in enumerate(frame_time):
            name = self.name_files[int(ft)]
            plot_test_img(pt_2d[num].detach().cpu().numpy().squeeze(),\
                os.path.join(self.test_2d_proj_path, '{}_plot_2d.png'.format(name)),\
                self.image_size[1],self.image_size[0],mask_d[num].detach().cpu().numpy().squeeze())
    def save_pt_clds(self,pt_cld_tensor,frame_time,rgb_color_ten):
        frame_time = frame_time.detach().cpu().numpy()

        for num,ft in enumerate(frame_time):
            name = self.name_files[int(ft)] 
            col = rgb_color_ten[0].detach().cpu().numpy().squeeze()*255
            save_ply_pt_clds(pt_cld_tensor[num].detach().cpu().numpy().squeeze(),\
                        file_name=os.path.join(self.test_pt_cld_path, '{}_pointCloud.ply'.format(name)),\
                        rgb_points = col.astype(np.uint8))

    def save_defromation_basis(self,deform_basis):
        deform_basis = deform_basis.squeeze().permute(0,2,1).cpu().numpy()
        num_basis,kps,_ = deform_basis.shape
        col = torch.rand((kps,3)).cpu().numpy().squeeze()*255
        for num,ft in enumerate(range(num_basis)):
            name = "basis_{:0>3d}".format(num) 
            
            save_ply_pt_clds(deform_basis[num].squeeze(),\
                        file_name=os.path.join(self.test_basis_pt_cld_path, '{}_pointCloud.ply'.format(name)),\
                        rgb_points = col.astype(np.uint8))        
    def save_test_image(self,img_tensor,frame_time,flag=None):
        frame_time = frame_time.detach().cpu().numpy().tolist()

        if flag == 'canonical':
            for num,ft in enumerate(frame_time):
                name = self.name_files[int(ft)]
                save_canonical_render(img_tensor[num].detach().cpu().numpy().squeeze(),\
                    os.path.join(self.test_canonical_path, '{}_cn_render.png'.format(name)))
        elif flag == 'color':
            for num,ft in enumerate(frame_time):
                name = self.name_files[int(ft)]    
                save_color_render(img_tensor[num].detach().cpu().numpy().squeeze(),\
                    os.path.join(self.test_colour_path, '{}_clr_render.png'.format(name)))       
        elif flag == 'depth':
            for num,ft in enumerate(frame_time):
                name = self.name_files[int(ft)] 
                self.max_v.append(img_tensor[num].detach().cpu().numpy().squeeze().max())
                self.min_v.append(img_tensor[num].detach().cpu().numpy().squeeze().min())
                save_depth_image(img_tensor[num].detach().cpu().numpy().squeeze(),\
                    os.path.join(self.test_depth_path, '{}_depth.png'.format(name)))
        elif flag == 'py3d_color':
            for num,ft in enumerate(frame_time):
                name = self.name_files[int(ft)]     
                save_color_render(img_tensor[num].detach().cpu().numpy().squeeze(),\
                    os.path.join(self.test_py3d_colour_path, '{}_mask_render.png'.format(name)))       
    def save_test_video(self,foldername, flag=None):
        if flag == 'ori':
            # print(foldername)
            filenames = sorted(glob.glob(os.path.join(foldername,'rgb','*.png')))
            filenames_mask = sorted(glob.glob(os.path.join(foldername,'mask','*.png')))
            filenames = filenames[:self.num_frames]
            filenames_mask = filenames_mask[:self.num_frames]
            
            vid_name  = os.path.join(self.test_video_path,'{}.avi'.format(flag))
            fourcc = cv.VideoWriter_fourcc(*'DIVX') 
            frame = cv.imread(list(filenames)[0])
            height, width, layers = frame.shape

            video = cv.VideoWriter(vid_name, fourcc, self.fps, (width, height))

            # Appending the images to the video one by one 
            for image,mask_ in zip(filenames,filenames_mask):
                mask = cv.imread(mask_)
                mask = np.where(mask==0,0,1)
                img = cv.imread(image)
                img = img*mask

                video.write(np.uint8(img))
                
            # Deallocating memories taken for window creation 
            cv.destroyAllWindows()
            video.release()  # releasing the video generated             

        elif flag == 'ori_depth':
            # print(foldername)
            filenames = sorted(glob.glob(os.path.join(foldername,'depth','*.png')))
            filenames_mask = sorted(glob.glob(os.path.join(foldername,'mask','*.png')))
            vid_name  = os.path.join(self.test_video_path,'{}.avi'.format(flag))
            fourcc = cv.VideoWriter_fourcc(*'DIVX') 
            frame = plt.imread(list(filenames)[0]) #plt.imread(image_file) cv.imread(
            height, width = frame.shape

            video = cv.VideoWriter(vid_name, fourcc, self.fps , (width, height))

            # Appending the images to the video one by one 
            for image,mask_ in zip(filenames,filenames_mask):
                mask = cv.imread(mask_)
                mask = np.where(mask==0,0,1)
                img = cv.imread(image)
                img = img*mask
                
                
                # print(img.shape)
                depth_norm = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
                depth_norm = 255*depth_norm
                heatmap = cv.applyColorMap(depth_norm.astype(np.uint8), cv.COLORMAP_JET)
                # print(np.unique(depth_norm))
                video.write(np.uint8(heatmap))
                # heatmap = cv.applyColorMap(depth_norm.astype(np.uint8), cv.COLORMAP_JET)
                # cv.imwrite(os.path.join(self.test_video_path,'tes.png'),cv.cvtColor(img.astype(np.uint8),cv.COLOR_BGR2RGB)) 
                # break
            # Deallocating memories taken for window creation 
            cv.destroyAllWindows()
            video.release()  # releasing the video generated  
            
        elif flag == 'canonical':
            filenames = sorted(glob.glob(os.path.join(foldername,'*.png')))
            vid_name  = os.path.join(self.test_video_path,'{}.avi'.format(flag))
            fourcc = cv.VideoWriter_fourcc(*'DIVX') 
            frame = cv.imread(list(filenames)[0])
            height, width, layers = frame.shape

            video = cv.VideoWriter(vid_name, fourcc, self.fps , (width, height))

            # Appending the images to the video one by one 
            for image in filenames:
                video.write(cv.imread(image))
                
            # Deallocating memories taken for window creation 
            cv.destroyAllWindows()
            video.release()  # releasing the video generated             
            
            
        elif flag == 'color':
            filenames = sorted(glob.glob(os.path.join(foldername,'*.png')))
            vid_name  = os.path.join(self.test_video_path,'{}.avi'.format(flag))
            fourcc = cv.VideoWriter_fourcc(*'DIVX') 
            frame = cv.imread(list(filenames)[0])
            height, width, layers = frame.shape

            video = cv.VideoWriter(vid_name, fourcc, self.fps , (width, height))

            # Appending the images to the video one by one 
            for image in filenames:
                video.write(cv.imread(image))
                
            # Deallocating memories taken for window creation 
            cv.destroyAllWindows()
            video.release()  # releasing the video generated 
        elif flag == 'depth':
            filenames = sorted(glob.glob(os.path.join(foldername,'*.png')))
            vid_name  = os.path.join(self.test_video_path,'{}.avi'.format(flag))
            fourcc = cv.VideoWriter_fourcc(*'DIVX') 
            frame = cv.imread(list(filenames)[0])
            height, width, layers = frame.shape

            video = cv.VideoWriter(vid_name, fourcc, self.fps , (width, height))

            # Appending the images to the video one by one 
            for image in filenames:
                video.write(cv.imread(image))
                
            # Deallocating memories taken for window creation 
            cv.destroyAllWindows()
            video.release()  # releasing the video generated 

        # elif flag == 'proj_2d':
        else:
            filenames = sorted(glob.glob(os.path.join(foldername,'*.png')))
            vid_name  = os.path.join(self.test_video_path,'{}.avi'.format(flag))
            fourcc = cv.VideoWriter_fourcc(*'DIVX') 
            frame = cv.imread(list(filenames)[0])
            height, width, layers = frame.shape

            video = cv.VideoWriter(vid_name, fourcc, self.fps , (width, height))

            # Appending the images to the video one by one 
            for image in filenames:
                video.write(cv.imread(image))
                
            # Deallocating memories taken for window creation 
            cv.destroyAllWindows()
            video.release()  # releasing the video generated 

    def save_gif_video(self,imgdir,flag=None ):
        from glob import glob
        # ffmpeg_boilerplate = '-y -f image2 -framerate {}'.format(self.fps)
        # ffmpeg_boilerplate += ' -hide_banner -loglevel error'
        nrows = 1
        ncols = 1
        dimx = self.image_size[1]
        dimy = self.image_size[0]
        # if flag == 'color':
        #     flag_2 = 'clr_render' 
        # elif flag == 'depth':
        #     flag_2 = 'depth'
        # elif flag == 'proj_2d':
        #     flag_2 = 'plot_2d'
        
        img_list = []
        file_imgs = sorted(glob(os.path.join(imgdir,'*.png')))
        for t_i in file_imgs:
            img = cv.imread(t_i)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_list.append(img) 

        # the .gif file can get pretty large
        imageio.mimsave(os.path.join(self.test_video_path,'{}.gif'.format(flag)), img_list, fps=self.fps)
if __name__ == '__main__':
    print('Inferencing Started') # very important!

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    logging.getLogger('PIL').setLevel(logging.WARNING) # avoids excessive logging by PIL::PngImagePlugin.py

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    parser = ArgumentParser(description="Training script parameters")

    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('-c','--conf', type=str, required=True,help='path to yaml config file')
    parser.add_argument('-w','--model_path', type=str,  required=True, help='path to output path')
    parser.add_argument('-d','--data_dir', type=str,  required=True, help='path to source data path')
    parser.add_argument('-i','--test_iteration', default=-1, type=int,help='iteration for which to extract pcds')
    parser.add_argument('-m','--mode', default='test', type=str,help='mode')
    parser.add_argument('-g','--gpu', default=0, type=int,help='index of gpu')
    args = parser.parse_args()
    
    npg_confs = os.path.join(args.conf)

    with open(npg_confs, 'r') as f:
        npg_confs = yaml.safe_load(f)
    npg_confs['data_dir'] = args.data_dir
    torch.cuda.set_device(args.gpu)

    runner = test_Runner(npg_confs, args.mode,args)

    if args.mode == 'test':
        # Test the model
        runner.test()