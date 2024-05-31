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
# import numba
# DEPTH_MAX, DEPTH_MIN = 2.4, 0.6
# DEPTH_VALID_MAX, DEPTH_VALID_MIN = 2.37, 0.63
# def read_depth_objaverse(depth_fn):
#     depth = imread(depth_fn)
#     depth = depth.astype(np.float32) / 65535 * (DEPTH_MAX-DEPTH_MIN) + DEPTH_MIN
#     mask = (depth > DEPTH_VALID_MIN) & (depth < DEPTH_VALID_MAX)
#     return depth, mask

# Read camera information from cameras.json
# with open('cameras.json', 'r') as f:
#     cameras = json.load(f)
# # Extract relevant information from cameras dictionary
# K = cameras['K']
# POSES = cameras['poses']
# # Convert K to numpy array
# K = np.array(K)
# # Convert POSES to numpy array
# POSES = np.array(POSES)

# K, _, _, _, POSES = read_pickle(f'meta_info/camera-16.pkl')
# H, W, NUM_IMAGES = 256, 256, 16
CACHE_DIR = './eval_mesh_pts'

def rasterize_depth_map(mesh,pose,K,shape):
    voxel_size = 0.1  # 设置体素大小
    mesh = mesh.voxel_down_sample(voxel_size)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    pts, depth = project_points(vertices,pose,K)
    # normalize to projection
    h, w = shape
    pts[:,0]=(pts[:,0]*2-w)/w
    pts[:,1]=(pts[:,1]*2-h)/h
    near, far = 5e-1, 1e2
    z = (depth-near)/(far-near)
    z = z*2 - 1
    pts_clip = np.concatenate([pts,z[:,None]],1)

    pts_clip = torch.from_numpy(pts_clip.astype(np.float32)).cuda()
    indices = torch.from_numpy(faces.astype(np.int32)).cuda()
    pts_clip = torch.cat([pts_clip,torch.ones_like(pts_clip[...,0:1])],1).unsqueeze(0)
    ctx = dr.RasterizeCudaContext(torch.device('cuda:0'))
    rast, _ = dr.rasterize(ctx, pts_clip, indices, (h, w)) # [1,h,w,4]
    depth = (rast[0,:,:,2]+1)/2*(far-near)+near
    mask = rast[0,:,:,-1]!=0
    # print(mask)
    return depth.cpu().numpy(), mask.cpu().numpy().astype(bool)

def ds_and_save(cache_dir, name, pts, cache=False):
    cache_dir.mkdir(exist_ok=True, parents=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    if cache:
        o3d.io.write_point_cloud(str(cache_dir/(name + '.ply')), downpcd)
    return downpcd

# {"id": 0, 
#  "img_name": "000", 
#  "width": 512, 
#  "height": 512, 
#  "position": [0.4720952027373465, -2.5167139839737094, 2.833305149252453], 
#  "rotation": [[-0.9916043347589784, 0.045389886428589994, -0.12108097083028291], [-0.09369221060767603, 0.3931574102691402, 0.9146852029096731], [0.0891213384024892, 0.9183501559647401, -0.38560391346676753]], 
#  "fy": 564.3795081665208, "fx": 562.4610037566981}
def load_cameras(cameras_fn):
    with open(cameras_fn, 'r') as f:
        cameras = json.load(f)
        num_images = len(cameras)
        W = cameras[0]['width']
        H = cameras[0]['height']
        K = np.array([[cameras[0]['fx'], 0, W/2], [0, cameras[0]['fy'], H/2], [0, 0, 1]])
        poses = []
        for cam in cameras:
            t = np.array(cam['position'])
            R = np.array(cam['rotation'])
            poses.append(K @ np.concatenate([R, t[:,None]], 1))
    poses = np.array(poses)
    return num_images, poses, K, H, W

# def get_points_from_mesh(mesh, name, num_images, poses, K, H, W, cache=False):
#     obj_name = name
#     cache_dir = Path(CACHE_DIR)
#     fn = cache_dir/f'{obj_name}.ply'
#     if cache and fn.exists():
#         pcd = o3d.io.read_point_cloud(str(fn))
#         return np.asarray(pcd.points)


#     pts = []
#     for index in range(num_images):
#         pose = poses[index]
#         depth, mask = rasterize_depth_map(mesh, pose, K, (H, W))
#         pts_ = mask_depth_to_pts(mask, depth, K)
#         pose_inv = pose_inverse(pose)
#         pts.append(pose_apply(pose_inv, pts_))

#     pts = np.concatenate(pts, 0).astype(np.float32)
#     downpcd = ds_and_save(cache_dir, obj_name, pts, cache)
    return np.asarray(downpcd.points,np.float32)

# @numba.jit(nopython=False)
def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    # n1, n2, v1, v2, tri_vert = torch.Tensor([n1]).cuda(), torch.Tensor([n2]).cuda(), torch.Tensor(v1).cuda(), torch.Tensor(v2).cuda(), torch.Tensor(tri_vert).cuda()
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def sample_single_tri_task(args):
    return sample_single_tri(*args)

def get_points_from_mesh(tsdf, thresh=0.2):
    vertices = np.asarray(tsdf.vertices)
    triangles = np.asarray(tsdf.triangles)
    tri_vert = vertices[triangles]
    v1 = tri_vert[:,1] - tri_vert[:,0]
    v2 = tri_vert[:,2] - tri_vert[:,0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:,0]
    # print("v1", v1.shape)
    # print("v2", v2.shape)
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)
    with mp.Pool(processes=8) as mp_pool:
        new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in tqdm(range(len(n1)))), chunksize=32)

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    return data_pcd


# def get_points_from_depth(depth_dir, obj_name):
#     cache_dir = Path(CACHE_DIR)
#     fn = cache_dir/f'{obj_name}.ply'
#     if fn.exists():
#         pcd = o3d.io.read_point_cloud(str(fn))
#         return np.asarray(pcd.points)

#     pts = []
#     for k in range(NUM_IMAGES):
#         depth, mask = read_depth_objaverse(os.path.join(depth_dir,f'{k:03}-depth.png'))
#         pts_ = mask_depth_to_pts(mask, depth, K)
#         pose_inv = pose_inverse(POSES[k])
#         pts.append(pose_apply(pose_inv, pts_))

#     pts = np.concatenate(pts, 0).astype(np.float32)
#     downpcd = ds_and_save(cache_dir, obj_name, pts, True)
#     return np.asarray(downpcd.points,np.float32)

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
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1.0 / resolution, min_bound=mesh.get_min_bound(), max_bound=mesh.get_max_bound()
    )

    # Initialize the 3D grid
    grid_shape = (resolution, resolution, resolution)
    voxel_data = np.zeros(grid_shape, dtype=bool)

    # Iterate through the voxels and set the corresponding grid positions to True
    for voxel in voxel_grid.get_voxels():
        grid_idx = voxel.grid_index
        if grid_idx[0]<resolution and grid_idx[1]<resolution and grid_idx[2]<resolution:
            voxel_data[grid_idx[0], grid_idx[1], grid_idx[2]] = True

    return voxel_data

def get_chamfer_iou(mesh_pr, mesh_gt, name, pr_type, gt_type, output, cameras_path, downsample,voxel_size=128):
    num_images, poses, K, H, W = load_cameras(cameras_path)
    pts_pr = get_points_from_mesh(mesh_pr, downsample)
    pts_gt = get_points_from_mesh(mesh_gt, downsample)
    # Save pts_pr and pts_gt as point clouds
    o3d.io.write_point_cloud(r"logs/pts_pr.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_pr)))
    o3d.io.write_point_cloud(r"logs/pts_gt.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_gt)))

    # compute iou
    size = voxel_size
    # sdf_pr = mesh2sdf.compute(mesh_pr.vertices, mesh_pr.triangles, size, fix=False, return_mesh=False)
    # sdf_gt = mesh2sdf.compute(mesh_gt.vertices, mesh_gt.triangles, size, fix=False, return_mesh=False)
    # vol_pr = sdf_pr<0
    # vol_gt = sdf_gt<0
    print('voxelization')
    vol_pr  = mesh_to_voxels(mesh_pr, resolution=64)
    vol_gt  = mesh_to_voxels(mesh_gt, resolution=64)    
    np.save(r'logs/vol_pr.npy', vol_pr)
    np.save(r'logs/vol_gt.npy', vol_gt)

    iou = np.sum(np.logical_and(vol_pr,vol_gt))/np.sum(np.logical_or(vol_gt, vol_pr))

    print("pts_pr", pts_pr.shape)
    print("pts_gt", pts_gt.shape)
    dist0 = nearest_dist(pts_pr, pts_gt, batch_size=4096)
    dist1 = nearest_dist(pts_gt, pts_pr, batch_size=512)
    print("dist0", np.mean(dist0))
    print("dist1", np.mean(dist1))

    chamfer = (np.mean(dist0) + np.mean(dist1)) / 2
    return chamfer, iou

def preprocess_mesh(mesh_pr, mesh_gt, mesh_downsample=True):
    if mesh_downsample:
        voxel_size_gt = max(mesh_gt.get_max_bound() - mesh_gt.get_min_bound()) 
        voxel_size_pr = max(mesh_pr.get_max_bound() - mesh_pr.get_min_bound())
        print(f'voxel_size of gt :{voxel_size_gt:e}')
        print(f'voxel_size of pr :{voxel_size_pr:e}')
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
    parser.add_argument('--pr_mesh', type=str, default=r"D:\wyh\eval_mvs\ec536168-0\ec536168-0\train\ours_30000\fuse_unbounded_post.ply")
    parser.add_argument('--pr_type', type=str, default="mesh")
    parser.add_argument('--gt_mesh', type=str, default=r"D:\wyh\eval_mvs\LEGO_Duplo_Build_and_Play_Box_4629\LEGO_Duplo_Build_and_Play_Box_4629\mesh\meshes\model.obj")
    parser.add_argument('--cameras_path', type=str, default=r".\cameras.json")
    parser.add_argument('--name', type=str, default="LEGO_Duplo_Build_and_Play_Box_4629")
    parser.add_argument('--gt_type', type=str, default="mesh")
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--mesh_downsample', action='store_true', default=True)
    parser.add_argument('--output', action='store_true', default=True, dest='output')
    args = parser.parse_args()

    mesh_gt = o3d.io.read_triangle_mesh(args.gt_mesh)
    vertices_gt = np.asarray(mesh_gt.vertices)
    mesh_gt.vertices = o3d.utility.Vector3dVector(vertices_gt)
    
    mesh_pr = o3d.io.read_triangle_mesh(args.pr_mesh)
    # vertices_pr = np.asarray(mesh_pr.vertices)
    # mesh_pr.vertices = o3d.utility.Vector3dVector(vertices_pr)
    mesh_pr = preprocess_mesh(mesh_pr, mesh_gt, args.mesh_downsample)
    vertices_pr = np.asarray(mesh_pr.vertices)
    mesh_pr.vertices = o3d.utility.Vector3dVector(vertices_pr)

    chamfer, iou = get_chamfer_iou(mesh_pr, mesh_gt, args.name, args.pr_type, args.gt_type, args.output, args.cameras_path, args.voxel_size)

    threshold, truncation_acc, truncation_com = 0.5, 2, 2
    _, dist_p = nn_correspondance(vertices_gt, vertices_pr, truncation_acc, True) # find nn in ground truth samples for each predict sample -> precision related
    _, dist_r = nn_correspondance(vertices_pr, vertices_gt, truncation_com, False) # find nn in predict samples for each ground truth sample -> recall related
    dist_p = np.array(dist_p)
    dist_r = np.array(dist_r)
    precision = np.mean((dist_p < threshold).astype('float')) * 100.0 # %
    recall = np.mean((dist_r < threshold).astype('float')) * 100.0 # %
    f_score = 2 * precision * recall / (precision + recall + 1e-8) # %

    results0 = f'case_name\t chamfer\t iou\t f_score'
    results = f'{args.name}\t {chamfer:.5f}\t {iou:.5f}\t {f_score:.5f}'
    # results = f'{args.pr_name}\t chamfer:{chamfer:.5f}\t iou:{iou:.5f}\t f_score:{f_score:.5f}'
    print(results0)
    print(results)
    with open('logs/metrics/mesh.log','a') as f:
        f.write(results+'\n')

if __name__=="__main__":
    main()