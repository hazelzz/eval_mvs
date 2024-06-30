import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
from skimage.morphology import binary_dilation, disk
import argparse
from tqdm import tqdm
import trimesh
from pathlib import Path
import transforms3d
import math


import eval_dtu.render_utils as rend_util

def norm_coords(vertices):
    max_pt = np.max(vertices, 0)
    min_pt = np.min(vertices, 0)
    scale = 1 / np.max(max_pt - min_pt)
    vertices = vertices * scale

    max_pt = np.max(vertices, 0)
    min_pt = np.min(vertices, 0)
    center = (max_pt + min_pt) / 2
    vertices = vertices - center[None, :]
    return vertices

def cull_scan(scan, mesh_path, result_mesh_file, instance_dataset):

    
    # load poses
    instance_dir = os.path.join(instance_dataset, f'scan{scan}')
    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())
    
    # load mask
    mask_dir = '{0}/mask'.format(instance_dir)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)

    # hard-coded image shape
    W, H = 1600, 1200

    # load mesh
    mesh = trimesh.load(mesh_path)
    
    vertices = mesh.vertices
    # vertices = norm_coords(vertices)
    R = transforms3d.euler.euler2mat(np.pi /2, 0, 0, 'szyx')
    vertices = vertices @ R.T

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    sampled_masks = []
    for i in tqdm(range(n_images)):
        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()
        intrinsic = intrinsics_all[i].cuda()

        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # dialate mask similar to unisurf
            maski = masks[i][:, :, 0].astype(np.float32) / 256.
            maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()
            
            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]
            sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)

    sampled_masks = torch.stack(sampled_masks, -1)
    # filter
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    
    # transform vertices to world 
    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh.export(result_mesh_file)
    del mesh
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )
    parser.add_argument('--scan_id', type=str,  help='scan id of the input mesh',default='24')
    parser.add_argument('--model_path', type=str,  help='model_path', default=r"D:\wyh\eval_mvs\ec536168-0\ec536168-0")
    parser.add_argument('--DTU', type=str,  help='path to the GT DTU point clouds', default=r"D:\wyh\dtu\SampleSet\SampleSet\MVS Data")
    parser.add_argument('--DTU_mask', type=str,  help='path to the DTU dataset', default=r"D:\wyh\dtu\dtu\DTU")
    args = parser.parse_args()

    out_dir = os.path.join(args.model_path, "eval_dtu")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ply_file = os.path.join(args.model_path, r"train\ours_30000\fuse_unbounded_post.ply")
    result_mesh_file = os.path.join(out_dir, r"culled_mesh.ply")
    cull_scan(args.scan_id, ply_file, result_mesh_file, args.DTU_mask)

    cmd = f"python D:\wyh\eval_mvs\eval_dtu\eval.py --data {result_mesh_file} --scan {args.scan_id} --mode mesh --dataset_dir {args.DTU} --vis_out_dir {out_dir}"
    print(cmd)
    os.system(cmd)
