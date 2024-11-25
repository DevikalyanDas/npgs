import warnings
warnings.filterwarnings("ignore")
import logging
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image
logging.getLogger('matplotlib.font_manager').disabled = True
import os
import glob,json
import numpy as np
import cv2 as cv
import numpy as np
import io
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from PIL import Image
from pathlib import Path
import struct
# import OpenEXR
# from visdom import Visdom

# from tools import NumpySeedFix

# the visdom connection handle
# viz = None

# class NumpySeedFix(object):

#     def __init__(self, seed=0):
#         self.rstate = None
#         self.seed = seed

#     def __enter__(self):
#         self.rstate = np.random.get_state()
#         np.random.seed(self.seed)

#     def __exit__(self, type, value, traceback):
#         if not(type is None) and issubclass(type, Exception):
#             print("error inside 'with' block")
#             return
#         np.random.set_state(self.rstate)

def plot_point_clouds(points,file_name=None):
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(-points[:, 0], points[:, 2], -points[:, 1],c = np.random.rand(len(points[:, 0]),3))
    ax.set_axis_off()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # ax1 = fig.add_subplot(112, projection="3d")
    # ax1.scatter(points[1, : ,0], points[1,:, 1], points[1,:, 2])
    # ax1.set_axis_off()
    plt.savefig(file_name)
    plt.show()


# def matplot_plot_point_cloud(ptcloud, pointsize=20, azim=90, elev=90,
#                              figsize=(8, 8), title=None, sticks=None, lim=None,
#                              cmap='gist_ncar', ax=None, subsample=None,
#                              flip_y=False,file_name= None):

#     if lim is None:
#         lim = np.abs(ptcloud).max()

#     nkp = int(ptcloud.shape[1])
#     pid = np.linspace(0., 1., nkp)
#     rgb = (cm.get_cmap(cmap)(pid)[:, :3]*255.).astype(np.int32)

#     if subsample is not None:
#         with NumpySeedFix():
#             prm = np.random.permutation(nkp)[0:subsample]
#         pid = pid[prm]
#         rgb = rgb[prm, :]
#         ptcloud = ptcloud[:, prm]

#     if flip_y:
#         ptcloud[1, :] = -ptcloud[1, :]

#     if ax is not None:
#         fig = None
#     else:
#         fig = plt.figure(figsize=figsize)
#         ax = fig.add_subplot(111, projection='3d')

#     ax.view_init(elev=elev, azim=azim)
#     if sticks is not None:
#         for stick in sticks:
#             line = ptcloud[:, [stick[0], stick[1]]]
#             xs, ys, zs = line
#             ax.plot(xs, ys, zs, color='black')

#     print(ptcloud.shape)
#     xs, ys, zs = ptcloud[:,0],ptcloud[:,1],ptcloud[:,2]
#     ax.scatter(xs, ys, zs, s=pointsize, c=pid, marker='.', cmap=cmap)

#     ax.set_xlim(-lim, lim)
#     ax.set_ylim(-lim, lim)
#     ax.set_zlim(-lim, lim)
#     ax.set_zticklabels([])
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     plt.axis('off')
#     if title is not None:
#         ax.set_title(title)
#     plt.savefig(file_name)
#     plt.show()
def save_ply_pt_clds(xyz_points,file_name=None,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'
    # xyz_points = bytes(xyz_points, "utf-8")
    # rgb_points = bytes(rgb_points, "utf-8")
    # Write header of .ply file
    fid = open(file_name,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file

    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tobytes(),rgb_points[i,1].tobytes(),
                                        rgb_points[i,2].tobytes())))
    fid.close()
def save_depth_image(depth,filename):
    # depth_norm = depth-depth.min()/(depth.max()-depth.min())
    # depth_norm = 255*depth_norm
    # back_gnd_idx = np.argwhere(depth_norm<0)
    
    # # print(np.unique(back_gnd_idx))
    
    # for i in back_gnd_idx:
    #     heatmap[tuple(i)]=0
    # print(np.unique(heatmap))
    # heatmap = cv.cvtColor(heatmap)

    # depth_norm = (depth-depth.min())/(1.0-depth.min())
    # heatmap = cv.applyColorMap(depth_norm.astype(np.uint8), cv.COLORMAP_JET)
    # heatmap = cv.cvtColor(heatmap)
    # depth[depth>0.6] = -1

    plt.imshow(depth,cmap='rainbow',vmax=0.5) #cmap='viridis' ,cmap='rainbow',
    plt.axis('off')
    plt.colorbar()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()
    # depth_norm = cv.normalize(depth, 0,255, norm_type=cv.NORM_MINMAX)
    # matplotlib.image.imsave(filename, depth,vmax=0.35,cmap='rainbow') #cmap=plt.cm.jet
    # cv.imwrite(filename,cv.cvtColor(depth,cv.COLOR_BGR2RGB)) 
    # cv.imwrite(filename,cv.cvtColor(heatmap,cv.COLOR_BGR2RGB))
def save_canonical_render(cn_render,filename):
    cn_render_norm = cv.normalize(cn_render, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    # print(cn_render_norm.min())
    # print(cn_render_norm.max())
    # back_gnd_idx = np.argwhere(cn_render)
    # cn_render = cv.normalize(cn_render, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    
    cn_render_norm = 255*cn_render_norm
    # back_gnd_idx = np.argwhere(cn_render_norm<0)
    # for i in back_gnd_idx:
    #     cn_render_norm[tuple(i)]=0
    
    # cn_render_norm[cn_render_norm.any()>111 and cn_render_norm.any()<112]=255

    cv.imwrite(filename,cv.cvtColor(cn_render_norm.astype(np.uint8),cv.COLOR_BGR2RGB))
    # matplotlib.image.imsave(filename, depth)  

def save_color_render(clr_render,filename):
    # plt.figure()
    # plt.imshow(clr_render) # Plot the image, turn off interpolation
    # # plt.gca.invert_xaxis()
    # # plt.gca.invert_yaxis()
    # plt.savefig(filename)
    # plt.show() # Show the image window
    # clr_render_norm = clr_render-clr_render.min()/(clr_render.max()-clr_render.min())

    clr_render_norm = cv.normalize(clr_render, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
 
    clr_render_norm = 255*clr_render_norm        
    cv.imwrite(filename,cv.cvtColor(clr_render_norm.astype(np.uint8),cv.COLOR_BGR2RGB))
    # matplotlib.image.imsave(filename, clr_render_norm)
    # plt.imshow(clr_render) #cmap='viridis' vmin=0.0, vmax=1.0,
    # plt.axis('off')
    # # plt.colorbar()
    # plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    # plt.close()

def plot_test_img(pts1,filename,w1,h1,mask):
    # return None

    contours, _ = cv.findContours(mask.squeeze().astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    img_back = np.zeros((h1,w1,3))
    img_back.fill(255) # or img[:] = 255
    img_back = cv.drawContours(img_back, contours, -1, (0, 0, 0), 2)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    # ax.imshow(img_back)
    # ax.plot(pts1[:1, 0] ,pts1[:1, 1],'.',color="red", alpha=1)
    
    ax.plot(pts1[1:, 0] ,pts1[1:, 1],'.',color="cyan", alpha=0.2)
    ax.scatter(pts1[:1, 0] ,pts1[:1, 1],c='black',s=50)
    # ax.plot(cand_pts[:, 0]*w1, cand_pts[:, 1]*h1,'+',color="red", alpha=0.3)

    plt.xlim([0, w1])
    plt.ylim([0,h1])
    plt.gca().invert_yaxis()    
    ax.set_axis_off()

    # ax1 = fig.add_subplot(112, projection="3d")
    # ax1.scatter(points[1, : ,0], points[1,:, 1], points[1,:, 2])
    # ax1.set_axis_off()
    plt.savefig(filename)
    plt.show()
    plt.close()
    # cv.imwrite(filename,cv.cvtColor(pts.astype(np.uint8),cv.COLOR_BGR2RGB))
# def plot_nerf_results(pred,filename,w1,h1.flag='depth'):
#     if flag=='depth':
#         # img_arr = np.zeros()
#         pred_img = pred.depth[0].squeeze(0).reshape((64,64))


#     img = 
def plot_overlay_pts_img(pts1,filename,w1,h1,img):
    # return None
    fig_w = 5
    fig_h = fig_w * (h1/w1)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.plot(pts1[:, 0] ,pts1[:, 1],'.',color="red", alpha=0.2)
    ax.set_axis_off()
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_deform_codes(deform_codes,frame_time,filename,json_name):
    xs = []
    ys = []
    coeff_def_dict = {}
    cols = len(frame_time)

    rows = deform_codes.shape[-1]
    for num,ft in enumerate(frame_time):
        coeff_def_dict['frame_{}'.format(ft)] = list(deform_codes[num].astype(float))
    json.dump(coeff_def_dict, open(json_name, 'w' ))
    for i,l in enumerate(frame_time):
        for j in range(0,rows):
            xs.append(l)
            ys.append(deform_codes[i,j])
    
    y_maxm = max(ys)
    y_minm = min(ys)
    mn_tm = frame_time.min()
    mx_tm = frame_time.max()    
    plt.scatter(xs,ys,marker='.',label='Deform_coeff')
        # plt.plot(i,j,'ok')

    # plt.plot(0,0,'ok') #<-- plot a black point at the origin
    # plt.axis('equal')  #<-- set the axes to the same scale
    plt.xlim([mn_tm-0.5,mx_tm+0.5]) #<-- set the x axis limits
    plt.ylim([y_minm-0.5,y_maxm+0.5]) #<-- set the y axis limits
    plt.legend() #<-- give a legend
    plt.grid(b=True, which='major') #<-- plot grid lines
    plt.xlabel('Frame Time')
    plt.ylabel('Values')
    plt.savefig(filename,bbox_inches='tight')

    plt.show()

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))