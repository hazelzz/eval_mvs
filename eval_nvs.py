import os

from PIL import Image
import cv2
import numpy as np
from argparse import ArgumentParser

import torch
from skimage.io import imread
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure
import lpips
from sam_utils import sam_init, sam_out_nosave
from util import pred_bbox, image_preprocess_nosave
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def compute_psnr_float(img_gt, img_pr):
    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)/ 255.0
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)/ 255.0
    mse = np.mean((img_gt - img_pr) ** 2, 0)
    mse = np.mean(mse) 
    psnr = 10 * np.log10(1 / mse)
    return psnr

def color_map_forward(rgb):
    rgb = np.array(rgb, dtype=np.float32)
    # print(rgb)
    dim = rgb.shape[-1]
    new_size = (777, 581)
    if dim==3:
        rgb=cv2.resize(rgb,new_size,interpolation=cv2.INTER_CUBIC)
        return np.array(rgb, dtype=np.float32) /255
    else:
        rgb = np.array(rgb, dtype=np.float32)
        rgb, alpha = rgb[:,:,:3], rgb[:,:,3:]
        rgb = rgb * alpha + (1-alpha)
        rgb=cv2.resize(rgb,new_size,interpolation=cv2.INTER_CUBIC)
        return np.uint8(rgb)

def preprocess_image(models, img_path, GT = False):
    img = Image.open(img_path)
    # if not img.mode == 'RGBA':
    img.thumbnail([777, 581], Image.Resampling.LANCZOS)
    #     img = sam_out_nosave(models['sam'], img.convert("RGB"), pred_bbox(img))
    #     torch.cuda.empty_cache()
    # else:
        # img = np.array(img, dtype=np.float32) / 255.0
        # img = img[:, :, 3:4] * img+ (1.0 - img[:, :, 3:4]) * np.ones_like(img)
        # img = img[:, :, :3]
        # img = Image.fromarray(img*255, mode='RGBA')
    img = sam_out_nosave(models['sam'], img.convert("RGB"), pred_bbox(img))
    torch.cuda.empty_cache()
    # img = image_preprocess_nosave(img, lower_contrast=False, rescale=True)
    return color_map_forward(img)
 

def main():
    parser = ArgumentParser()
    parser.add_argument('--gt',type=str,default=r'D:\wyh\eval_mvs\dtu122_2dgs\test\ours_30000\gt')
    parser.add_argument('--pr',type=str,default=r'D:\wyh\SuGaR\rendered_images\pr')
    parser.add_argument('--name',type=str, default=r'DTU_bird')
    parser.add_argument('--num_images',type=int, default=8)
    args = parser.parse_args()

    num_images = args.num_images
    gt_dir= args.gt
    pr_dir = args.pr

    models = {}
    models['sam'] = sam_init(0, r"ckpts\sam_vit_h_4b8939.pth")
    models['lpips'] = lpips.LPIPS(net='vgg').cuda().eval()
    fid_score = 0

    psnrs, ssims, lpipss, l1losses = [], [], [], []
    for k in tqdm(range(num_images)):
        img_gt = preprocess_image(models,os.path.join(gt_dir, f'{k:05}.png'),True)
        img_pr = preprocess_image(models,os.path.join(pr_dir, f'{k:05}.png'),False)

        save_path = os.path.join(pr_dir,'preprocessed')
        os.makedirs(save_path, exist_ok=True)
        Image.fromarray((img_gt*255).astype(np.uint8)).save(os.path.join(save_path,f'gt_{k:05}.png'))
        Image.fromarray((img_pr*255).astype(np.uint8)).save(os.path.join(save_path,f'pr_{k:05}.png'))
        # img_gt.save(os.path.join(save_path,f'gt_{k:05}.png'))
        # img_pr.save(os.path.join(save_path,f'pr_{k:05}.png'))
        psnr = compute_psnr_float(img_gt, img_pr)

        with torch.no_grad():
            img_gt_tensor = torch.from_numpy(img_gt.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
            img_pr_tensor = torch.from_numpy(img_pr.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
            ssim = float(structural_similarity_index_measure(img_pr_tensor, img_gt_tensor).flatten()[0].cpu().numpy())
            gt_img_th, pr_img_th = img_gt_tensor*2-1, img_pr_tensor*2-1
            score = float(models['lpips'](gt_img_th, pr_img_th).flatten()[0].cpu().numpy())
            l1loss = np.mean(np.abs(img_gt, img_pr))

        ssims.append(ssim)
        lpipss.append(score)
        psnrs.append(psnr)
        l1losses.append(l1loss)


    msg0=f'case_name\t      psnrs\t     ssims\t     lpipss\t    l1losses'
    msg=f'{args.name}\t {np.mean(psnrs):.5f}\t {np.mean(ssims):.5f}\t {np.mean(lpipss):.5f}\t {np.mean(l1losses):.5f}'
    print(msg0)
    print(msg)
    with open('logs/metrics/nvs.log','a') as f:
        f.write(msg0+'\n')
        f.write(msg+'\n')

if __name__=="__main__":
    main()