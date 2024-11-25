import numpy as np
import cv2 as cv
import random
import torch
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import yaml
import os

class NumpySeedFix(object):

    def __init__(self, seed=0):
        self.rstate = None
        self.seed = seed

    def __enter__(self):
        self.rstate = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, type, value, traceback):
        if not(type is None) and issubclass(type, Exception):
            print("error inside 'with' block")
            return
        np.random.set_state(self.rstate)

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
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Random_Seq_Batch_Sampler(Sampler):
    def __init__(self, dataset,bs=4):
        self.perm_len = int(len(dataset))
        self.lim = list(range(self.perm_len-bs + 1))
        self.bs = bs
        self.cnt = self.perm_len-self.bs+1

    def __iter__(self):

        while True:
            elem = random.randrange(0,self.cnt, 1)
            lis = list(range(elem,elem+self.bs))
            yield lis
    
    def __len__(self):
        return len(self.perm_len)

class Fixed_Seq_Batch_Sampler(Sampler):
    def __init__(self, dataset,bs=4):
        self.perm_len = int(len(dataset))
        self.lim = list(range(self.perm_len-bs + 1))
        self.bs = bs
        self.cnt = self.perm_len-self.bs+1
        self.start = 0

    def __iter__(self):

        while True:
            elem = self.start % self.cnt
            self.start = self.start + 1
            lis = list(range(elem,elem+self.bs))
            yield lis

    def __len__(self):
        return len(self.perm_len)
        
class Random_Incre_Batch_Sampler(Sampler):
    def __init__(self, dataset,bs=4, it=450):
        self.perm_len = int(len(dataset))
        self.lim = list(range(self.perm_len))
        self.bs = bs
        self.it = it
    def __iter__(self):
        count = 0
        check_dup = set()
        while count <= self.it:
            
            lis = sorted(random.sample(self.lim, self.bs))
            count+=1
            
            yield lis
    
    def __len__(self):
        return len(self.perm_len)

# class Stored_Random_Seq_Batch_Sampler(Sampler):
#     def __init__(self, dataset,bs=4):
#         self.perm_len = int(len(dataset))
#         self.bs = bs
#         self.cnt = self.perm_len-self.bs+1
#         self.start = 0
#         self.all_elements = list(range(self.perm_len))
#         self.all_data = list(self.all_elements[i:i+bs] for i in range(self.cnt))
#         self.poping_list = None
#     def __iter__(self):

#         while True:
#             # elem = random.randrange(0,self.cnt, 1)
#             if not self.poping_list:
#                 self.poping_list = self.all_data.copy()
#                 random.shuffle(self.poping_list)
#             yield self.poping_list.pop()

#     def __len__(self):
#         return self.perm_len
class Stored_Random_Seq_Batch_Sampler(Sampler):
    def __init__(self, dataset,bs=4):
        self.perm_len = int(len(dataset))
        self.bs = bs
        self.cnt = self.perm_len-self.bs+1
        self.start = 0
        self.all_elements = list(range(self.perm_len))
        self.all_data = list(self.all_elements[i:i+bs] for i in range(self.cnt))
        self.pop_data = list(range(self.cnt))
        self.poping_list = None
    def __iter__(self):

        while True:
            if not self.poping_list:
                self.poping_list = self.pop_data.copy()
                random.shuffle(self.poping_list)                
            yield self.all_data[self.poping_list.pop()]

    def __len__(self):
        return self.perm_len

def warp_back_gs(x,flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    flo = flo.permute(0 ,2 ,3 ,1)
    output = F.grid_sample(x, vgrid,align_corners=True,mode='bicubic')
    mask = (torch.ones(x.size())).cuda()
    mask = F.grid_sample(mask, vgrid,mode='bicubic', align_corners=True)

    mask[mask <0.9999] = 0
    mask[mask >0] = 1

    return output*mask

def custom_collate_fn(original_batch):
    img_list,mask_list,ref_fg_list,flow_list,frame_time_list,proj_mat_list,candidate_loc_list,\
    fg_points_list,intrinsics_list,pose_list,frame_time_nov,ref_prj_mat,fg_lengths_list = [],[],[],[],[],[],[],[],[],[],[],[],[]

    for it in original_batch:
        img_list.append(it[0])
        mask_list.append(it[1])
        ref_fg_list.append(it[2])
        flow_list.append(it[3])
        frame_time_list.append(it[4])
        proj_mat_list.append(it[5])
        candidate_loc_list.append(it[6])
        fg_points_list.append(it[7])
        intrinsics_list.append(it[8])
        pose_list.append(it[9])
        frame_time_nov.append(it[10])
        
        ref_prj_mat.append(it[11])
        fg_lengths_list.append(it[12])

    fg_points_list1 =  torch.nn.utils.rnn.pad_sequence(fg_points_list, batch_first=True, padding_value=10000.0)
    
    return torch.stack(img_list),torch.stack(mask_list),torch.stack(ref_fg_list),\
        torch.stack(flow_list),torch.stack(frame_time_list),torch.stack(proj_mat_list),\
            torch.stack(candidate_loc_list),fg_points_list1,torch.stack(intrinsics_list),\
            torch.stack(pose_list),torch.stack(frame_time_nov),torch.stack(ref_prj_mat),torch.stack(fg_lengths_list)

def fps(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

import traceback

def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)

def dump_yaml(path, cfgs, cfg_file = None):
    print(f"Saving configs to {path}")
    xmkdir(os.path.dirname(path))
    with open(os.path.join(path,cfg_file), 'w') as outfile:
        return yaml.dump(cfgs, outfile, default_flow_style=False, sort_keys=False)
"""
A simple writer for .ply geometry files. Very useful for debugging!
Buffers the contents and writes out only on exit (necessary for writing
the header without knowing the vertex/edge/face count in advance).
Closes the fd in the case a with block raises an exception.
We could wait until the exit to even open the fd; however, I decided
against that to match expected behaviour. This could accomodate a
design where the vertex/edge/face count is provided in advance.

Usage:
```
with PLYWriter('some_file.ply`, hasEdges=True) as f:
    if len(start_pts) != len(end_pts):
        raise ValueError('must have end for each start!')
    for v in start_pts:
        f.addPoint(v)
    for v in end_pts:
        f.addPoint(v)
    for i in range(len(start_pts)):
        f.addEdge(i, i + len(start_pts))
```
"""

class PLYWriter:
    def __init__(
        self,
        filename,
        hasNormals=False,
        hasColours=False,
        hasEdges=False,
        hasFaces=False
    ):
        self.filename = filename
        if not filename.endswith('.ply'):
            self.filename += '.ply'
        self.hasNormals = hasNormals
        self.hasColours = hasColours
        self.hasEdges = hasEdges
        self.hasFaces = hasFaces
        self.vertices = []
        self.edges = []
        self.faces = []

    def __enter__(self):
        self.fd = open(self.filename, 'w')
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            self.fd.close()
            return False # carry exception on
        # Now we can write out everything!
        self.__writeHeader()
        for vert in self.vertices:
            self.fd.write(vert)
            self.fd.write('\n')
        for edge in self.edges:
            self.fd.write(edge)
            self.fd.write('\n')
        for face in self.faces:
            self.fd.write(face)
            self.fd.write('\n')
        self.fd.close()
        return True

    def addPoint(self, pos, normal=None, colour=None):
        vert = ''
        if len(pos) != 3:
            raise ValueError('invalid position length')
        vert += '{:.6f} {:.6f} {:.6f}'.format(pos[0], pos[1], pos[2])
        if self.hasNormals:
            if len(normal) != 3:
                raise ValueError('invalid normal length')
            vert += ' {:.6f} {:.6f} {:.6f}'.format(normal[0], normal[1], normal[2])
        else:
            if normal != None:
                raise ValueError('unexpected normal provided!')
        if self.hasColours:
            if len(colour) != 3:
                raise ValueError('invalid colour length')
            vert += ' {:.0f} {:.0f} {:.0f}'.format(colour[0], colour[1], colour[2])
        else:
            if colour != None:
                raise ValueError('unexpected colour provided!')
        self.vertices.append(vert)
    
    def addEdge(self, v1, v2):
        if not self.hasEdges:
            print('unexpected edge provided! setting hasEdges to True')
            self.hasEdges = True
        self.edges.append('{} {}'.format(v1, v2))

    def addFace(self, v1, v2, v3, v4=None):
        if not self.hasFaces:
            print('unexpected face provided! setting hasFaces to True')
            self.hasFaces = True
        if v4 is not None:
            self.faces.append('4 {} {} {} {}'.format(v1, v2, v3, v4))
        else:
            self.faces.append('3 {} {} {}'.format(v1, v2, v3))

    def __writeHeader(self):
        self.fd.write('ply\n')
        self.fd.write('format ascii 1.0\n')
        self.fd.write('comment File generated by PLYWriter v0.6.10\n')
        self.fd.write('element vertex {}\n'.format(len(self.vertices)))
        self.fd.write('property float x\n')
        self.fd.write('property float y\n')
        self.fd.write('property float z\n')
        if self.hasNormals:
            self.fd.write('property float nx\n')
            self.fd.write('property float ny\n')
            self.fd.write('property float nz\n')
        if self.hasColours:
            self.fd.write('property uchar red\n')
            self.fd.write('property uchar green\n')
            self.fd.write('property uchar blue\n')
        if self.hasEdges:
            self.fd.write('element edge {}\n'.format(len(self.edges)))
            self.fd.write('property int vertex1\n')
            self.fd.write('property int vertex2\n')
        if self.hasFaces:
            self.fd.write('element face {}\n'.format(len(self.faces)))
            self.fd.write('property list uchar int vertex_indices\n')
        self.fd.write('end_header\n')