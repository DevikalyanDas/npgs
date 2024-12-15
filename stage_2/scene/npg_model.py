import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


# from utils.rigid_utils import exp_se3
from plyfile import PlyData, PlyElement
import torch.nn as nn
import torch
import numpy as np

import random
import math
from glob import glob
from utils.system_utils import mkdir_p
from pathlib import Path

# from utilities.raft_demo import raft_pretrained

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True,num_frm = None
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
    tensor (torch.Tensor): Input tensor to be positionally encoded.
    encoding_size (optional, int): Number of encoding functions used to compute
        a positional encoding (default: 6).
    include_input (optional, bool): Whether or not to include the input in the
        positional encoding (default: True)

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    # Now, encode the input using a set of high-frequency functions and append the
    # resulting values to the encoding.
    for i in range(num_encoding_functions):
        for func in [torch.sin, torch.cos]:
          encoding.append(func(2. ** i * tensor)) #*math.pi
    return torch.cat(encoding, dim=-1)


class NPG(torch.nn.Module):

    def __init__(self,conf_args):
        super(NPG, self).__init__()

        # autoassign constructor params to self
        # auto_init_args(self)
        self.n_keypoints = int(conf_args['key_points_no'])
        self.shape_basis_size = int(conf_args['shape_basis_size'])
        self.color_basis_size = int(conf_args['color_basis_size'])

        self.n_fully_connected = int(conf_args['n_fully_connected'])

        self.n_layers = int(conf_args['n_layers'])
        self.vector_dim = int(conf_args['phi_vector_dim'])
        self.batch_size = 4
        self.num_pos_enc = int(conf_args['num_positional_encoding'])
        self.initial_dim = 1+2*self.num_pos_enc
        
        self.factor = int(conf_args['norm_data'])
        
        self.device = torch.device('cuda')



        self.psi = nn.Sequential(
                        *self.make_trunk(dim_in=self.initial_dim,
                        n_fully_connected=self.n_fully_connected,
                        n_layers=self.n_layers))

        # deformation coefficient predictor
        self.alpha_layer = conv1x1(self.n_fully_connected, self.shape_basis_size,std=0.01)
        # Color coefficient predictor
        self.beta_layer = conv1x1(self.n_fully_connected, self.color_basis_size,std=0.01)
        
        # Sigmoid of the color prediction (eqn-5)
        self.sigmoid_lyr = nn.Sigmoid()
        # Same for all frames. Defining the deformaton basis and color basis.
        #  b_dim = bs x basis_size X 3 X n_keypoints
        # self.deformation_basis = torch.rand(size=(conf.get_int('train.batch_size'),self.shape_basis_size, 3,self.n_keypoints),device=self.device,requires_grad =True) 
        # For color basis
        # self.color_basis = torch.rand(size=(conf.get_int('train.batch_size'),self.color_basis_size, 3, self.n_keypoints),device=self.device,requires_grad=True) 

    def make_trunk(self,
                    n_fully_connected=None,
                    dim_in=None,
                    n_layers=None,
                    use_bn=True):

        layer1 = ConvBNLayer(dim_in,
                                n_fully_connected,
                                use_bn=use_bn)
        layers = [layer1]

        for l in range(n_layers):
            layers.append(ResLayer(n_fully_connected,
                                    int(n_fully_connected/4)))

        return layers
    
    def forward(self, frame_time = None,deformation_basis=None,img_size=(480,640),num_frames = None):
        
        # dictionary with outputs of the fw pass
        preds = {}
        self.image_size = img_size

        factor = self.factor
        # Our input is the frame time. Runs the deformation coeff
        # ref_fr_rnd = random.choice(self.reference_frame)

        new_frame = frame_time.expand(self.batch_size) #torch.cat((frame_time,torch.tensor([self.reference_frame])),dim=0)

        frame_time_norm = torch.add(new_frame,1)/factor#num_frames #self.num_frames#/100    new_frame/(self.num_frames-1)#torch.add(new_frame,1)/100#frame_time*(2.0/self.num_frames)-1.0 #torch.add(frame_time,1)/100 #

        mlp_output = self.run_psi(positional_encoding(frame_time_norm[:,None],num_encoding_functions=self.num_pos_enc,num_frm=num_frames))

        # MLP output has both branches shape coefficient and color coefficient
        # coeff dim = bs X basis_size X 1 X 1
        def_coeff = mlp_output['deform_coeff'][:,:,None,None]  #dim=bs x deform_basis_size x1x1
        clr_coeff = mlp_output['color_coeff'][:,:,None,None] #dim=bs x color_basis_size x1x1

        ######coefficients interpolation #########
        # frame_time_1 = torch.tensor([19]).cuda()
        # frame_time_2 = torch.tensor([138]).cuda()

        # new_frame_1 = frame_time_1.expand(self.batch_size) #torch.cat((frame_time,torch.tensor([self.reference_frame])),dim=0)
        # new_frame_2 = frame_time_2.expand(self.batch_size)
        # frame_time_norm_1 = torch.add(new_frame_1,1)/factor#num_frames #self.num_frames#/100    new_frame/(self.num_frames-1)#torch.add(new_frame,1)/100#frame_time*(2.0/self.num_frames)-1.0 #torch.add(frame_time,1)/100 #
        # frame_time_norm_2 = torch.add(new_frame_2,1)/factor

        # mlp_output_1 = self.run_psi(positional_encoding(frame_time_norm_1[:,None],num_encoding_functions=self.num_pos_enc,num_frm=num_frames))
        # mlp_output_2 = self.run_psi(positional_encoding(frame_time_norm_2[:,None],num_encoding_functions=self.num_pos_enc,num_frm=num_frames))
        # # MLP output has both branches shape coefficient and color coefficient
        # # coeff dim = bs X basis_size X 1 X 1
        # def_coeff_1 = mlp_output_1['deform_coeff'][:,:,None,None]  #dim=bs x deform_basis_size x1x1
        # clr_coeff_1 = mlp_output_1['color_coeff'][:,:,None,None] #dim=bs x color_basis_size x1x1

        # def_coeff_2 = mlp_output_2['deform_coeff'][:,:,None,None]  #dim=bs x deform_basis_size x1x1
        # clr_coeff_2 = mlp_output_2['color_coeff'][:,:,None,None] #dim=bs x color_basis_size x1x1

        # interpolation_factor = 0.75
        # def_coeff = def_coeff_1*(1.0-interpolation_factor) + def_coeff_2*(interpolation_factor)
        ##########################################
        
        # creation of the 3D Keypoints. Remember to get the dimension of the basis changed to
        # 1 such that they can be added

        # deformation_basis = deformation_basis.repeat(self.batch_size,1,1,1)
        # color_basis = color_basis.repeat(self.batch_size,1,1,1)
        # print(deformation_basis.shape)
        # print(preds['deform_coeff'].shape)

        prod = torch.mul(deformation_basis,def_coeff)
        kp_3D = torch.sum(prod,dim=1)  # dimension

        # key_points_3D = kp_3D[:-1]  #Give all elements except the last (reference frame)
        preds['keypoint_3D'] = kp_3D[0].permute(1,0) # dim = bsx3xnum_keypts
               
        return preds

    def run_psi(self, frame_time):

        preds = {}

        # batch size
        ba = self.batch_size # no of frames
        # dtype = frame_time.type()

        # Input to be given to the network

        l1_input = frame_time

        # pass to network
        feats = self.psi(l1_input[:, :, None, None])

        # Shape coefficients into the linear basis
        shape_coeff = self.alpha_layer(feats)[:, :, 0, 0]
        preds['deform_coeff'] = shape_coeff

        # Color coefficients into the linear basis ## not required
        color_coeff = self.beta_layer(feats)[:, :, 0, 0]
        preds['color_coeff'] = color_coeff

        return preds

    def norm_points(self,kp_cloud):
        # kp_cloud = (B+1) X 3 X kp_num
        
        centroid = torch.mean(kp_cloud,2)
        kp_cloud -= centroid[...,None]
        scale = torch.max(torch.sqrt(torch.sum(torch.abs(kp_cloud)**2,dim=1)),dim=1)[0][...,None,None]
        kp_cloud = kp_cloud/scale

        return kp_cloud

    def descriptors(self,bs,vec_len):
        # convert to torch tensors
        # finding a number between -1,1 (from a cube) and normalize them to make each vector unit length
        fin = []
        for _ in range(bs):
            arr = np.array([[random.uniform(-1.0, 1.0) for i in range(self.vector_dim)] for j in range(vec_len)])
            desc_features = np.array([arr[i,:]/np.linalg.norm(arr[i,:],axis=0) for i in range(vec_len)])
            fin.append(desc_features)
        bs_des = torch.Tensor(np.array(fin))
        return bs_des  

# def pytorch_ge12():
#     v = torch.__version__
#     v = float('.'.join(v.split('.')[0:2]))
#     return v >= 1.2

def conv1x1(in_planes, out_planes, std=0.01):
    """1x1 convolution"""
    cnv = nn.Conv2d(in_planes, out_planes, bias=True, kernel_size=1)

    cnv.weight.data.normal_(0., std)
    if cnv.bias is not None:
        cnv.bias.data.fill_(0.)

    return cnv


class ConvBNLayer(nn.Module):

    def __init__(self, inplanes, planes, use_bn=False, stride=1, ):
        super(ConvBNLayer, self).__init__()

        # do a reasonable init
        self.conv1 = conv1x1(inplanes, planes)
        self.use_bn = use_bn
        # if use_bn:
        #     self.bn1 = nn.BatchNorm2d(planes)
            # if pytorch_ge12():
            #     self.bn1.weight.data.uniform_(0., 1.)
        self.relu = nn.LeakyReLU(inplace=False)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        # if self.use_bn:
        #     out = self.bn1(out)
        out = self.relu(out)
        return out


class ResLayer(nn.Module):

    def __init__(self, inplanes, planes, expansion=4):
        super(ResLayer, self).__init__()
        self.expansion = expansion

        self.conv1 = conv1x1(inplanes, planes)
        # self.bn1 = nn.BatchNorm2d(planes)
        # if pytorch_ge12():
        #     self.bn1.weight.data.uniform_(0., 1.)
        self.conv2 = conv1x1(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        # if pytorch_ge12():
        #     self.bn2.weight.data.uniform_(0., 1.)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # if pytorch_ge12():
        #     self.bn3.weight.data.uniform_(0., 1.)
        self.relu = nn.LeakyReLU(inplace=False)
        self.skip = inplanes == (planes*self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.skip:
            out += residual
        out = self.relu(out)

        return out

class NPGModel:
    def __init__(self, conf_args,source_path,checkpoint_path):
        self.npg_model = NPG(conf_args).cuda()
        #checkpoint_path = glob(os.path.join(source_path,'npg_weight','*.pth'))[0]

        checkpoint = torch.load(checkpoint_path)
        print('Stage I weight found at iteration: {}'.format(Path(checkpoint_path).stem.split('_')[-1]))
        self.npg_model.load_state_dict(checkpoint['model_network'])
        self._deformation_basis = checkpoint['deformation_basis'].cuda().requires_grad_(False)
        self.optimizer = None
        self.spatial_lr_scale = 1.0

    def step(self,frame_time = None,img_size=(480,640),total_frames=None):
        return self.npg_model(frame_time, self._deformation_basis, img_size, total_frames)  # size: num_kps x 3

    def train_setting(self, training_args):
        self._deformation_basis = nn.Parameter(self._deformation_basis.requires_grad_(True))
        l = [
            {'params': list(self.npg_model.parameters()),'lr': training_args.npg_lr_init * self.spatial_lr_scale,"name": "npg"},
            {'params': [self._deformation_basis],'lr': training_args.deform_basis_lr_init * self.spatial_lr_scale,"name": "deform_basis"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.npg_scheduler_args = get_expon_lr_func(lr_init=training_args.npg_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.npg_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.npg_lr_delay_mult,
                                                    max_steps=training_args.npg_lr_max_steps)
    def update_npg_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "npg":
                lr = self.npg_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path,"npg_weight", "iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        checkpoint = {
                'model_network':self.npg_model.state_dict(),
                'deformation_basis':self._deformation_basis.detach().cpu(),
        }
        torch.save(checkpoint, os.path.join(out_weights_path, 'npg_all.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "npg_weight"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "npg_weight/iteration_{}/npg_all.pth".format(loaded_iter))
        checkpoint = torch.load(weights_path)
        self.npg_model.load_state_dict(checkpoint['model_network'])
        self._deformation_basis = checkpoint['deformation_basis'].cuda()

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        # l.append('opacity')
        # for i in range(self._scaling.shape[1]):
        #     l.append('scale_{}'.format(i))
        # for i in range(self._rotation.shape[1]):
        #     l.append('rot_{}'.format(i))

        # for i in range(self._lrf.shape[2]):
        #     l.append('lrf_rot_{}'.format(i))
        return l

    def save_ply(self, path, xyz, ids):
        
        Path(path).mkdir(parents=True, exist_ok=True)
        # tot_frames = len(self._xyz)
        # for idx in range(tot_frames):
        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        # scale = self._scaling.detach().cpu().numpy()
        # rotation = self._rotation.detach().cpu().numpy()
        # lrf = self._lrf[idx].cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, lrf), axis=1)
        attributes = np.concatenate((xyz, normals), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        file_name = os.path.join(path, "{:0>8d}_pointCloud.ply".format(ids))
        PlyData([el]).write(file_name)
        
        

