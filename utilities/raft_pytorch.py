import sys
sys.path.append('/BS/keytr_neus/work/devi_work/code/utilities')
# sys.path.append('/BS/keytr_neus/work/devi_work/code/RAFT/core')
# sys.path.append('/BS/keytr_neus/work/devi_work/code/RAFT/models')


import matplotlib.pyplot as plt
import torchvision.transforms.functional as Fu
# from torchvision.models.optical_flow import Raft_Large_Weights
# from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# from raft import RAFT
# from flow_viz import flow_to_image


# weights = Raft_Large_Weights.DEFAULT
# transforms = weights.transforms()

device = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = 'cuda'

# def load_image(imfile):
#     img = np.array(Image.open(imfile)).astype(np.uint8)
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#     return img[None].to(DEVICE)

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

def viz(img,flo, id2, img2, warp):
    
    #frame_img = warp_back_gs(img2,flo)
    img = img[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo1 = flo[0].permute(1,2,0).cpu().numpy()
    warp = warp[0].permute(1,2,0).cpu().numpy()
    # Remap the second to first

    #frame_img = wrap_back(img2,flo)

    diff = img - warp
    #save the images for visualization
    cv2.imwrite('first_{}.png'.format(id2), cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    cv2.imwrite('warped_{}.png'.format(id2), cv2.cvtColor(warp,cv2.COLOR_BGR2RGB))
    cv2.imwrite('difference_{}.png'.format(id2), cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
    cv2.imwrite('second_{}.png'.format(id2), cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo1)
   
    
    cv2.imwrite('flow_field_{}.png'.format(id2), cv2.cvtColor(flo,cv2.COLOR_BGR2RGB))    
    
    img_flo = np.concatenate([img, flo], axis=0)
    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    #cv2.waitKey()

def raft_pytorch(image_batch,render_j,masks_bat,path_save,fr_time,itr,flow_batch):

    # model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    # model = model.eval()
    id1 = 0
    # warped_final = []

    
        
    render_2 = render_j[1:]
    if render_2.shape[1] != 3:
        render_2 = render_2.permute(0,3,1,2)
    flow_pred = flow_batch[:-1]

    warped_final = warp_back_gs(render_2,flow_pred)

    with torch.no_grad():    
        
        if itr[1]%4000==0:
            warped_2_1 = warped_final[0].clone()
            warped_2_1 *= masks_bat[0]
            image1 = image_batch[0] * masks_bat[0]
            flow_img = flow_to_image(flow_pred[0]).detach().cpu().permute(1,2,0).numpy()
            cv2.imwrite(os.path.join(path_save,'warp_{}_{}.png'.format(itr[0],fr_time[0].detach().cpu().numpy())),cv2.cvtColor(((warped_2_1*255).squeeze().permute(1,2,0).detach().cpu().numpy()).astype(np.uint8),cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(path_save,'flow_{}_{}.png'.format(itr[0],fr_time[0].detach().cpu().numpy())),cv2.cvtColor((flow_img).astype(np.uint8),cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(path_save,'first_img_{}_{}.png'.format(itr[0],fr_time[0].detach().cpu().numpy())),cv2.cvtColor((image1.squeeze().permute(1,2,0).detach().cpu().numpy()).astype(np.uint8),cv2.COLOR_BGR2RGB))

    return warped_final
        # for num, (frame_i, frame_j, cn_render_j,masks_i,masks_j) in enumerate(zip(image_batch[:-1], \
        #                             image_batch[1:],render_j[1:],masks_bat[:-1],masks_bat[1:])):
            

        #     image1,image2,render_2,mask_1,mask_2 = frame_i[None,:,:,:], frame_j[None,:,:,:], cn_render_j[None,:,:,:],masks_i[None,:,:,:],masks_j[None,:,:,:]
            

        #     img1_t,img2_t = transforms(image1.type(torch.uint8),image2.type(torch.uint8))
        #     # print('+++++++++++++')
        #     # print(img1_t.detach().cpu().numpy())

        #     list_of_flows = model(img1_t.to(device), img2_t.to(device))
            
        #     predicted_flow = list_of_flows[-1].squeeze()

            
        #     flow_img = flow_to_image(predicted_flow).detach().cpu().permute(1,2,0).numpy()
        #     # predicted_fl_np = predicted_flow.detach().cpu().permute(1,2,0).numpy()

        #     # arr = np.save(os.path.join(save_path,'{}.npy'.format(file_n)), predicted_fl_np)            
        #     if render_2.shape[1] != 3:
        #         render_2 = render_2.permute(0,3,1,2)
        #     # predicted_flow = predicted_flow[None,...] * mask_1

        #     # render_2 *= mask_2
        #     warped_2_1 = warp_back_gs(render_2,predicted_flow[None,...])
        #     warped_final.append(warped_2_1.squeeze())
        #     warped_2_1 *= mask_1
        #     image1 *= mask_1
        #     if itr[1]%4000==0:
        #         cv2.imwrite(os.path.join(path_save,'warp_{}_{}.png'.format(itr[0],fr_time[num].detach().cpu().numpy())),cv2.cvtColor(((warped_2_1*255).squeeze().permute(1,2,0).detach().cpu().numpy()).astype(np.uint8),cv2.COLOR_BGR2RGB))
        #         cv2.imwrite(os.path.join(path_save,'flow_{}_{}.png'.format(itr[0],fr_time[num].detach().cpu().numpy())),cv2.cvtColor((flow_img).astype(np.uint8),cv2.COLOR_BGR2RGB))
        #         cv2.imwrite(os.path.join(path_save,'first_img_{}_{}.png'.format(itr[0],fr_time[num].detach().cpu().numpy())),cv2.cvtColor((image1.squeeze().permute(1,2,0).detach().cpu().numpy()).astype(np.uint8),cv2.COLOR_BGR2RGB))
            
        #     # print(warped_2_1.shape)
        #     #for visualization
        #     # viz(image1, flow_up, id1,image2,warped_2_1)
        #     # id1 += 1
        #     # result = torch.cat(warped_final,dim=0)


        # return torch.stack(warped_final)