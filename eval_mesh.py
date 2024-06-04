import argparse
import os
# from pathlib import Path

import torch
import numpy as np
import transforms3d
# from skimage.io import imread
from tqdm import tqdm

from ldm.base_utils import project_points, mask_depth_to_pts, pose_inverse, pose_apply, output_points, read_pickle
import open3d as o3d
import mesh2sdf
import json
import nvdiffrast.torch as dr
import multiprocessing as mp
import eval_dtu.evaluate_single_scene as eval_dtu
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor



def nearest_dist(pts0, pts1, batch_size=128):
    pts0 = torch.from_numpy(pts0.astype(np.float32)).cuda()
    pts1 = torch.from_numpy(pts1.astype(np.float32)).cuda()
    pn0, pn1 = pts0.shape[0], pts1.shape[0]
    dists = []
    for i in tqdm(range(0, pn0, batch_size), desc='evaluating...'):
        dist = torch.norm(pts0[i:i+batch_size,None,:] - pts1[None,:,:], dim=-1)
        dists.append(torch.min(dist,1)[0])
    dists = torch.cat(dists,0)
    return dists.cpu().numpy()

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

def transform_gt(vertices, rot_angle):
    vertices = norm_coords(vertices)
    R = transforms3d.euler.euler2mat(-np.deg2rad(rot_angle), 0, 0, 'szyx')
    vertices = vertices @ R.T

    return vertices

def mesh_to_voxels(mesh, resolution=32):
    # Get the voxel grid
    scale = 1 / np.max(mesh.get_max_bound() - mesh.get_min_bound())
    center=mesh.get_center()
    mesh.scale(scale, center)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1.0 / resolution)

    # Initialize the 3D grid
    grid_shape = (resolution, resolution, resolution)
    voxel_data = np.zeros(grid_shape, dtype=bool)


    def process_voxel(voxel):
        grid_idx = voxel.grid_index
        if grid_idx[0] < resolution and grid_idx[1] < resolution and grid_idx[2] < resolution:
            voxel_data[grid_idx[0], grid_idx[1], grid_idx[2]] = True

    with ThreadPoolExecutor() as executor:
        executor.map(process_voxel, tqdm(voxel_grid.get_voxels()))
    mesh.scale(1 / scale, center)

    return voxel_data

def get_iou(mesh_pr, mesh_gt, out_dir):
    # compute iou
    print('voxelization')
    vol_pr  = mesh_to_voxels(mesh_pr, resolution=128)
    vol_gt  = mesh_to_voxels(mesh_gt, resolution=128)    
    np.save(os.path.join(out_dir,r'vol_pr.npy'), vol_pr)
    np.save(os.path.join(out_dir,r'vol_gt.npy'), vol_gt)

    iou = np.sum(np.logical_and(vol_pr,vol_gt))/np.sum(np.logical_or(vol_gt, vol_pr))
    print("[iou]:", iou)
    return iou


def preprocess_mesh(mesh_pr, mesh_gt, mesh_downsample=True):
    if mesh_downsample:
        voxel_size_gt = max(mesh_gt.get_max_bound() - mesh_gt.get_min_bound()) 
        voxel_size_pr = max(mesh_pr.get_max_bound() - mesh_pr.get_min_bound())
        print(f'volume_size of gt :{voxel_size_gt:e}')
        print(f'volume_size of pr :{voxel_size_pr:e}')
        scale = voxel_size_gt / voxel_size_pr
        voxel_size = voxel_size_pr*scale
        mesh_downsample = mesh_pr.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)
        mesh_pr = mesh_downsample

    # Load bbox and mask
    # bbox = np.load('bbox.npy')
    # mask = np.load('mask.npy')

    # Apply bbox and mask to mesh
    # mesh_pr = apply_bbox_and_mask(mesh_pr, bbox, mask)

    return mesh_pr
def nn_correspondance(verts1, verts2, truncation_dist, ignore_outlier=True):
    """ for each vertex in verts2 find the nearest vertex in verts1
    Args:
        nx3 np.array's
        scalar truncation_dist: points whose nearest neighbor is farther than the distance would not be taken into account
    Returns:
        ([indices], [distances])
    """

    indices = []
    distances = []
    dist_ls = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    truncation_dist_square = truncation_dist**2

    for vert in verts2:
        _, inds, dist_square = kdtree.search_knn_vector_3d(vert, 1)
        dist_ls = np.append(dist_ls, np.sqrt(dist_square[0]))
        if dist_square[0] < truncation_dist_square:
            indices.append(inds[0])
            distances.append(np.sqrt(dist_square[0]))
        else:
            if not ignore_outlier:
                indices.append(inds[0])
                distances.append(truncation_dist)

    return indices, distances

# python eval_mesh.py --pr_mesh eval_examples/chicken-pr.ply --pr_name chicken --gt_dir eval_examples/chicken-gt --gt_mesh eval_examples/chicken-mesh/meshes/model.obj --gt_name chicken
# python eval_mesh.py --pr_mesh D:\2d-gaussian-splatting\output\ec536168-0\train\ours_30000\fuse_unbounded_post.ply --name LEGO_Duplo_Build_and_Play_Box_4629 --pr_type tsdf  --gt_mesh D:\Free3D\MVS_data\render_res\LEGO_Duplo_Build_and_Play_Box_4629\mesh\meshes\model.obj --gt_type mesh --cameras_path D:\2d-gaussian-splatting\output\ec536168-0\cameras.json --output 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr_mesh', type=str, default=r"D:\wyh\eval_mvs\dtu122\train\ours_30000\fuse_post.ply")
    parser.add_argument('--pr_dir', type=str, default=r"D:\wyh\eval_mvs\dtu122")
    parser.add_argument('--pr_type', type=str, default="mesh")
    parser.add_argument('--gt_mesh', type=str, default=r"D:\wyh\eval_mvs\dtu_data\MVS_dataset")
    parser.add_argument('--gt_mesh_colmap', type=str, default=r"D:\wyh\eval_mvs\dtu_data")
    parser.add_argument('--gt_mesh_mask', type=str, default=r"D:\wyh\eval_mvs\dtu_data\mask\scan122.ply")
    # parser.add_argument('--name', type=str, default="LEGO_Duplo_Build_and_Play_Box_4629")
    # parser.add_argument('--gt_type', type=str, default="mesh")
    parser.add_argument('--threshold', type=float, default=600)
    parser.add_argument('--downsample', action='store_true', default=False)
    # parser.add_argument('--output', action='store_true', default=True, dest='output')
    args = parser.parse_args()

    # preprocess mesh
    print("preprocessing mesh")
    ply_file = os.path.join(args.pr_mesh)
    out_dir = os.path.join(args.pr_dir, "eval_dtu")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pr_mesh_path = os.path.join(out_dir, r"culled_mesh.ply")
    gt_mesh_path = os.path.join(args.gt_mesh_mask)
    eval_dtu.cull_scan(122, ply_file, pr_mesh_path, args.gt_mesh_colmap)

    mesh_gt = o3d.io.read_triangle_mesh(gt_mesh_path)
    vertices_gt = np.asarray(mesh_gt.vertices)
    mesh_gt.vertices = o3d.utility.Vector3dVector(vertices_gt)
    
    mesh_pr = o3d.io.read_triangle_mesh(pr_mesh_path)
    vertices_pr = np.asarray(mesh_pr.vertices)
    mesh_pr.vertices = o3d.utility.Vector3dVector(vertices_pr)

    voxel_size_gt = max(mesh_gt.get_max_bound() - mesh_gt.get_min_bound()) 
    voxel_size_pr = max(mesh_pr.get_max_bound() - mesh_pr.get_min_bound())
    print(f'voxel_size of gt :{voxel_size_gt:e}')
    print(f'voxel_size of pr :{voxel_size_pr:e}')
    print("mesh_pr", vertices_pr.shape)
    print("mesh_gt", vertices_gt.shape)

    if args.downsample:
        scale = 0.01
        mesh_downsample = mesh_pr.simplify_vertex_clustering(
            voxel_size=voxel_size_pr*scale,
            contraction=o3d.geometry.SimplificationContraction.Average)
        mesh_pr = mesh_downsample
        mesh_downsample = mesh_gt.simplify_vertex_clustering(
            voxel_size=voxel_size_gt*scale,
            contraction=o3d.geometry.SimplificationContraction.Average)
        mesh_gt = mesh_downsample
        vertices_pr = np.asarray(mesh_pr.vertices)
        vertices_gt = np.asarray(mesh_gt.vertices)
        print("mesh_pr", vertices_pr.shape)
        print("mesh_gt", vertices_gt.shape)

    print("computing chamfer and f_score")
    cmd = f"python .\eval_dtu\eval.py --data {pr_mesh_path} --scan {122} --mode {args.pr_type} --dataset_dir {args.gt_mesh} --vis_out_dir {out_dir}"
    print(cmd)
    os.system(cmd)

    print("computing iou")
    iou = get_iou(mesh_pr, mesh_gt, out_dir)

    # threshold = args.threshold # how to set this threshold?
    # print("mesh to point cloud")
    # pts_pr_o3d = o3d.geometry.PointCloud()
    # pts_pr_o3d.points = o3d.utility.Vector3dVector(vertices_pr)
    # pts_gt_o3d = o3d.geometry.PointCloud()
    # pts_gt_o3d.points = o3d.utility.Vector3dVector(vertices_gt)
    # pts_pr_o3d = mesh_pr.sample_points_uniformly(number_of_points=10000)
    # pts_gt_o3d = mesh_gt.sample_points_uniformly(number_of_points=10000)
    # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.005)

    # print("computing f_score")
    # dist_p = pts_pr_o3d.compute_point_cloud_distance(pts_gt_o3d)
    # dist_r = pts_gt_o3d.compute_point_cloud_distance(pts_pr_o3d)
    # recall = float(sum(d < threshold for d in dist_p)) / float(len(dist_p))
    # precision = float(sum(d < threshold for d in dist_r)) / float(len(dist_r))
    # f_score = 2 * precision * recall / (precision + recall + 1e-8) # %
    
    # print("f_score", f_score)
    # print("precision", precision)
    # print("recall", recall)

if __name__=="__main__":
    main()