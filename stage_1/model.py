import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as Fu
import pytorch3d
import random
import math

from pytorch3d.structures import Pointclouds

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

# def mlp_alpha(pos_enc_time):
#   layers = nn.ModuleList(
#             [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

class NPG(torch.nn.Module):

    def __init__(self,conf,device):
        super(NPG, self).__init__()

        # autoassign constructor params to self
        # auto_init_args(self)
        self.n_keypoints = int(conf['key_points_no'])
        self.shape_basis_size = int(conf['shape_basis_size'])
        self.color_basis_size = int(conf['color_basis_size'])

        # self.weight_init_std = conf.get_int('model.weight_init_std') 
        self.n_fully_connected = conf['n_fully_connected']

        self.n_layers = conf['n_layers']
        self.vector_dim = conf['phi_vector_dim']
        self.batch_size = conf['batch_size']
        self.descriptor_dim = conf['phi_vector_dim']
        self.num_pos_enc = conf['num_positional_encoding']
        self.initial_dim = 1+2*self.num_pos_enc

        # self.case_data_pose = conf.get_string('dataset.case_pose')

        self.reference_frame = float(conf['ref_frame'])
        # self.reference_frame = conf.get_list('train.ref_frame')

        self.device=device

        self.num_frames = conf['num_frames']


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
        self.time_norm = conf['norm_data']#
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
    
    def forward(self, frame_time = None,proj_mat=None,deformation_basis=None,\
                            feat_descriptors=None,img_size=(480,640)):
        
        # dictionary with outputs of the fw pass
        preds = {}
        self.image_size = img_size
        
        # Our input is the frame time. Runs the deformation coeff
        # ref_fr_rnd = random.choice(self.reference_frame)

        new_frame = torch.cat((frame_time,torch.tensor([self.reference_frame])),dim=0)

        frame_time_norm = torch.add(new_frame,1)/self.time_norm
        
        mlp_output = self.run_psi(positional_encoding(frame_time_norm[:,None],num_encoding_functions=self.num_pos_enc,num_frm=self.num_frames))

        # MLP output has both branches shape coefficient and color coefficient
        # coeff dim = bs X basis_size X 1 X 1
        def_coeff = mlp_output['deform_coeff'][:,:,None,None]  #dim=bs x deform_basis_size x1x1
        clr_coeff = mlp_output['color_coeff'][:,:,None,None] #dim=bs x color_basis_size x1x1

        # creation of the 3D Keypoints. Remember to get the dimension of the basis changed to
        # 1 such that they can be added

        prod = torch.mul(deformation_basis,def_coeff)
        kp_3D = torch.sum(prod,dim=1)  # dimension


        key_points_3D = kp_3D[:-1]  #Give all elements except the last (reference frame)
        preds['keypoint_3D'] = key_points_3D # dim = bsx3xnum_keypts
        
        preds['deform_coeff'] = def_coeff[:-1]
        preds['color_coeff'] = clr_coeff[:-1]

        preds['kp_dist_frame'] = kp_3D[-1][None,...].repeat(self.batch_size,1,1) #(reference frame)
        # Turninig them into homogenous coordinates by appending a 1 to XYZ   # bs X 4 X 500
        bsize,_,num_keypts = key_points_3D.shape
        # dim_ones = torch.ones((bsize,1,self.n_keypoints))
        homo_key_pts_3D = Fu.pad(key_points_3D, (0, 0,0,1), "constant", 1.0) #torch.cat((key_points_3D,dim_ones),dim=1)
        
        # Finding the 2D projection of the key points
        proj_2d_homo_key = torch.bmm(proj_mat, homo_key_pts_3D)  # bs X 3 X 500
        # proj_2d_homo_key = key_points_3D
        z_depth = proj_2d_homo_key[:,2:3,:]  
        z_depth = torch.clamp(z_depth,0.1) # prevent division by zero. Next use this instead of clamp: + 0.0000001
        proj_2d_keypts = proj_2d_homo_key[:,0:2,:]/(z_depth)  # perspective projection


        # proj_2d_homo_key = proj_2d_keypts/focal[:, :, None]
        preds['proj_2d_keypts'] = proj_2d_keypts.permute(0,2,1)#[...,[1,0]] #dim = bs x num_pts x 2
        

        phi_features = feat_descriptors.expand(bsize,num_keypts,self.descriptor_dim) #self.descriptors(bsize, num_keypts) #dim = bs x num_keypts x feature_dim


        pt_cld_canonical_batch = Pointclouds(points=key_points_3D.permute(0,2,1), features=phi_features)
        preds['pt_cld_canon'] = pt_cld_canonical_batch
  

        preds['depth'] = z_depth
       
        return preds

    def run_psi(self, frame_time):

        preds = {}

        # batch size
        ba = self.batch_size # no of frames
        # dtype = frame_time.type()

        # Input to be given to the network
        
        # frame_tens = frame_time.reshape(ba,self.initial_dim)
        l1_input = frame_time  #torch.cat((kp_loc_flatten, kp_vis_in), dim=1)
        # print(l1_input.shape)
        # pass to network
        feats = self.psi(l1_input[:, :, None, None])

        # Shape coefficients into the linear basis
        shape_coeff = self.alpha_layer(feats)[:, :, 0, 0]
        preds['deform_coeff'] = shape_coeff

        # Color coefficients into the linear basis
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
