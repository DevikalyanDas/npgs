import torch
# from chamferdist import ChamferDistance
import torch.nn as nn
import pytorch3d
import numpy as np
import cv2 as cv
import random
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import look_at_view_transform, BlendParams, \
                                PointsRasterizationSettings, PointsRenderer, PulsarPointsRenderer,\
                                PointsRasterizer, AlphaCompositor,\
                                NormWeightedCompositor
from pytorch3d.loss import chamfer_distance
# from utilities.raft_demo import raft_pretrained
from utilities.raft_pytorch import raft_pytorch
from utilities.tools import warp_back_gs
# from utilities.chamfer_3d import chamfer_distance
from pytorch3d.utils import cameras_from_opencv_projection

# from tools import decompose_proj_matrix

# Non-Negative depth loss
class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super(ChamferDistanceLoss,self).__init__()
        self.test = 1.0
        self.w_xy = float(1.0)
        self.w_yx = float(1.0)
    def forward(self,fg_points,prj_points, x_lengths=None):
        loss_list = []
        fg_points = fg_points.float()
        prj_points = prj_points.float()
        x_lengths = x_lengths.long()
        for n,(fg,prj) in enumerate(zip(fg_points,prj_points)):
            fg_s = fg.squeeze()
            # print(fg_s)
            prj_s = prj.squeeze()
            condition = fg_s != 10000.0
            row_cond = condition.all(1)
            fg_s = fg_s[row_cond, :]
            # res = chamfer_distance(fg_s[None,...],prj_s[None,...])
            loss_cd,_ = chamfer_distance(fg_s[None,...],prj_s[None,...])
            # loss_cd = self.w_xy * res['cham_xy'] + self.w_yx * res['cham_yx']
            loss_list.append(loss_cd)

        
        # loss_cd,_  = chamfer_distance(fg_points,prj_points,x_lengths=x_lengths)
        # return loss_cd
        # ref_fr_sum = ref_fr_cd['cham_xy'] + ref_fr_cd['cham_yx']
        # loss_list.append(ref_fr_cd)
        
        return torch.mean(torch.stack(loss_list))

# Stolen from https://gist.github.com/wohlert/8589045ab544082560cc5f8915cc90bd
# Remember it takes the location into consideration for computing the distance
# The x and y going into forward takes the coordinates (the x,y coordinates of the points) 
# for two images x and y. Hence, for our case x will be batch_size * x_coord * y_coord for the
# points. The number of points in image x and y neednot be same as we are measuring transport
# This is the assignment loss
class SinkhornSolver(nn.Module):
    """
    Optimal Transport solver under entropic regularisation.
    Based on the code of Gabriel Peyré.
    """
    def __init__(self, epsilon, iterations=100, ground_metric=lambda x: torch.pow(x, 2)):
        super(SinkhornSolver, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.ground_metric = ground_metric

    def forward(self, x, y):
        num_x = x.size(-2)
        num_y = y.size(-2)
        
        batch_size = 1 if x.dim() == 2 else x.size(0)

        # Marginal densities are empirical measures
        a = x.new_ones((batch_size, num_x), requires_grad=False) / num_x
        b = y.new_ones((batch_size, num_y), requires_grad=False) / num_y
        
        a = a.squeeze()
        b = b.squeeze()
                
        # Initialise approximation vectors in log domain
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)

        # Stopping criterion
        threshold = 1e-1
        
        # Cost matrix
        C = self._compute_cost(x, y)
        
        # Sinkhorn iterations
        for i in range(self.iterations): 
            u0, v0 = u, v
                        
            # u^{l+1} = a / (K v^l)
            K = self._log_boltzmann_kernel(u, v, C)
            u_ = torch.log(a + 1e-8) - torch.logsumexp(K, dim=2)
            u = self.epsilon * u_ + u
                        
            # v^{l+1} = b / (K^T u^(l+1))
            K_t = self._log_boltzmann_kernel(u, v, C).transpose(-2, -1)
            v_ = torch.log(b + 1e-8) - torch.logsumexp(K_t, dim=2)
            v = self.epsilon * v_ + v
            
            # Size of the change we have performed on u
            diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
            mean_diff = torch.mean(diff)
                        
            if mean_diff.item() < threshold:
                break
   
        # print("Finished computing transport plan in {} iterations".format(i))
    
        # Transport plan pi = diag(a)*K*diag(b)
        K = self._log_boltzmann_kernel(u, v, C)
        pi = torch.exp(K)
        
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        return torch.mean(cost) #, pi

    def _compute_cost(self, x, y):
        x_ = x.unsqueeze(-2)
        y_ = y.unsqueeze(-3)
        C = torch.sum(self.ground_metric(x_ - y_), dim=-1)
        return C

    def _log_boltzmann_kernel(self, u, v, C=None):
        # C = self._compute_cost(x, y) if C is None else C
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= self.epsilon
        return kernel



class KPNegDepthLoss(nn.Module):
    def __init__(self):
        super(KPNegDepthLoss,self).__init__()
        """
        kp_pos: [B, num_sou, num_kp, 3]
        extr: [B, num_sou, num_tar, ]
        """

    def forward(self,keypt,extr):
        kp_pos = self.to_cam(keypt.unsqueeze(1), extr)      # [B, num_views, num_kp, 3]
        neg_depth = torch.clamp(kp_pos[..., 2], max=0)
        res = torch.mean(neg_depth ** 2)
        return res
 
    def to_cam(self,x,extr):
        x_hom = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        x_cam = (x_hom @ extr.transpose(-1, -2))[..., :3]
        return x_cam


class KPDistanceLoss(nn.Module):
    def __init__(self,conf,pr_fr=None):
        super(KPDistanceLoss,self).__init__()
        """
        kp_pos: [B, num_kp, 3]
        
        """
        self.k_nn = int(conf['k_nn'])
        self.bs = int(conf['batch_size'])
        
        #Added later for the pt_cld ref   
        # self.pr_frame_dis = pr_fr
        # _,self.pr_frame_ind = torch.topk(self.pr_frame_dis, k = self.k_nn,largest=False,dim=2)
        # self.dis_mat_knn_fixed = torch.gather(self.pr_frame_dis , dim=2, index = self.pr_frame_ind)
    def forward(self,keypt,fixed_frame):

        dis_mat_fixed = torch.cdist(fixed_frame,fixed_frame)**2#self.pairwise_dist(keypt,keypt)   torch.cdist(keypt,keypt)    # [B, num_kp, num_kp]
        dis_mat_kpt = torch.cdist(keypt,keypt)**2

        _,nn_dist_ind_fixed= torch.topk(dis_mat_fixed, k = self.k_nn,largest=False,dim=2)
        ## dis_mat_1 = nn_dist_val[0].expand(nn_dist_val.shape)

        dis_mat_knn_kpt = torch.gather(dis_mat_kpt , dim=2, index = nn_dist_ind_fixed)

        dis_mat_knn_fixed = torch.gather(dis_mat_fixed , dim=2, index = nn_dist_ind_fixed)

        final_dist = (dis_mat_knn_fixed-dis_mat_knn_kpt)**2
             
        return torch.sum(final_dist,2).mean()


def safe_sqrt(A, eps_1=float(1e-6)):
    """
    performs safe differentiable sqrt
    """

    return (torch.clamp(A,min=float(0)) + eps_1).sqrt()

def soft_huber(dfsq, eps):
    loss = eps * (safe_sqrt(1.0 + dfsq)-1.0)
    return loss

def soft_pt_rasterizer(pt_cloud,intrinsic,pose,img_size_t,image_size,radius,pts_per_pixel,flag=None):

    R, T = pose[:,:3,:3], pose[:,:3,3:4].squeeze()

    intrinsics= intrinsic[:,:3,:3]

    # point_cloud = Pointclouds(points=pt_cloud.squeeze(), features=feat)
    
    cameras = cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=intrinsics,\
                            image_size=img_size_t)
    
    # blend_params = BlendParams(sigma=1e-5, gamma=1e-4)
    raster_settings = PointsRasterizationSettings(
                image_size=image_size, 
                radius = radius, #np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                points_per_pixel = pts_per_pixel
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    

    renderer = PointsRenderer(
                rasterizer=rasterizer,
                compositor=AlphaCompositor())        
    
    images = renderer(pt_cloud)
    pt_fragments = None #rasterizer(pt_cloud)
   
    return images[:,:,:,:3], pt_fragments#.zbuf

class Flow_Consistency_Loss(nn.Module):
    def __init__(self,conf, device, save_path=None,bs = None,img_size=None, epsilon=0.01):
        super(Flow_Consistency_Loss,self).__init__()
        self.epsilon = float(epsilon)
        self.device = device
        self.conf = conf

        self.radius = float(conf['radius'])
        self.pts_per_pixel = int(conf['pts_per_pixel'])
        self.image_size = img_size
        self.img_size_tensor = torch.from_numpy(np.array(img_size))

        self.img_size_t = torch.stack([self.img_size_tensor for _ in range(bs)])
        self.save_path = save_path
    def forward(self,input_batch_out1, pt_cld_canon, flow_batch, mask_batch,intrinsic,pose,frame_time,itr):
        # vec_dim = self.conf.get_int('model.phi_vector_dim')

        canon_rendered_batch,depth_frag = soft_pt_rasterizer(pt_cld_canon,intrinsic,pose,self.img_size_t, \
            self.image_size,self.radius,self.pts_per_pixel)

        
        wrapped_batch1 = raft_pytorch(input_batch_out1,canon_rendered_batch,mask_batch.type(torch.uint8),self.save_path,frame_time,itr,flow_batch)

        x,y,mask_i = canon_rendered_batch[:-1],wrapped_batch1,mask_batch[:-1]

        diff = (x.permute(0,3,1,2) * mask_i) - (y * mask_i)

        bs,h,w,ch = x.shape

        diff = diff.reshape((bs,-1))
        mask_sum = mask_i.reshape((bs,-1))
        mask_sum = torch.clamp(mask_sum.sum(1), 1.)

        dist_hub =  soft_huber((diff**2)/(self.epsilon**2),eps=self.epsilon)
        
        fin_val = (dist_hub.sum(1)/mask_sum).mean()  #/mask_sum
        
        depth_fin = None#depth_frag.zbuf
 

        return fin_val,canon_rendered_batch,depth_fin #torch.mean(torch.tensor(final_loss))

 
# The RGB color loss for matching the RGB value assigned to each keypoint. (Not used currently)
class RGB_Loss(nn.Module):
    def __init__(self,conf, device,bs = None,img_size=None, epsilon=0.01):
        super(RGB_Loss, self).__init__()
        self.epsilon = float(epsilon)
        self.conf = conf
        self.device = device
        self.radius = float(conf['radius'])
        self.pts_per_pixel = int(conf['pts_per_pixel'])
        self.image_size = img_size
        self.img_size_tensor = torch.from_numpy(np.array(img_size))

        self.img_size_t = torch.stack([self.img_size_tensor for _ in range(bs)])
 
    def forward(self,input_batch_out1,pt_cld_clr, mask_batch, intrinsic, pose):
        
        final_loss = []
        # # create the point cloud data structure of pytorch3D
        # pt_cld_i = Pointclouds(points=key_pt_batch, features=color_features)        
        color_render_batch1,depth_frag = soft_pt_rasterizer(pt_cld_clr,intrinsic,pose,self.img_size_t, \
            self.image_size,self.radius,self.pts_per_pixel)
        # Normalizing the rendered image in the range 0-255
        # print(np.max(input_batch_out1.detach().cpu().numpy()))
        # print(np.min(input_batch_out1.detach().cpu().numpy()))
        # print(np.max(color_render_batch1.detach().cpu().numpy()))
        # print(np.min(color_render_batch1.detach().cpu().numpy()))
        bs,h,w,ch = color_render_batch1.shape
        color_render_batch = torch.reshape(color_render_batch1,(bs, -1))
        color_render_batch -= color_render_batch.min(1, keepdim=True)[0]
        color_render_batch /= (color_render_batch.max(1, keepdim=True)[0] + 1e-6)
        color_render_batch = torch.reshape(color_render_batch,(bs, h, w,ch))
        # color_render_batch = color_render_batch * 255.
        # input_batch_out = nn.functional.normalize(input_batch_out)
        bs1,ch1,h1,w1 = input_batch_out1.shape
        input_batch_out = torch.reshape(input_batch_out1,(bs1, -1))
        input_batch_out -= input_batch_out.min(1, keepdim=True)[0]
        input_batch_out /= input_batch_out.max(1, keepdim=True)[0]
        input_batch_out = torch.reshape(input_batch_out,(bs1,ch1, h1, w1))
        # print('+++++++++++++++++')
        # print(np.max(input_batch_out.detach().cpu().numpy()))
        # print(np.min(input_batch_out.detach().cpu().numpy()))
        # print(np.max(color_render_batch.detach().cpu().numpy()))
        # print(np.min(color_render_batch.detach().cpu().numpy()))
        x,y,mask_i = input_batch_out,color_render_batch,mask_batch #[1:]
        # print(torch.any(x.isnan()),torch.any(y.isnan()),torch.any(mask_i.isnan()))
        diff = (y.permute(0,3,1,2) * mask_i) - (x * mask_i)
        # bt,_,_,_ = diff.shape
        diff = diff.reshape((bs1,-1))
        
        dist_hub =  soft_huber((diff**2)/(self.epsilon**2),eps=float(self.epsilon))

        mask_sum = mask_i.reshape((bs,-1))
        mask_sum = mask_sum.sum(1)

        fin_val = (dist_hub.sum(1)/mask_sum).mean() # /mask_sum
        
        # dists2 = depth_frag.dists.permute(0,3,1,2)
        # weights = 1-dists2/(self.radius*self.radius)
        # depth_comp = AlphaCompositor()(depth_frag.idx.long().permute(0,3,1,2),weights,pt_cld_clr.features_packed().permute(1, 0))#depth_fin.zbuf.permute(0,3,1,2))

        # depth_fin = depth_comp.permute(0,2,3,1)#[:,:,:,0]#[:,:,:,None] 
        depth_fin = depth_frag.zbuf

        return fin_val,color_render_batch1,depth_fin #torch.mean(torch.tensor(final_loss))
