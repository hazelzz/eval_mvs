# Evaluation of Multi-View Stereo (MVS) Methods

## Introduction
This document provides instructions for evaluating Multi-View Stereo (MVS) methods. MVS is a computer vision technique that reconstructs a 3D scene from multiple 2D images taken from different viewpoints.
## Methods to evaluate
- [ ] [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://github.com/hbb1/2d-gaussian-splatting)
- [ ] [SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering](https://github.com/Anttwo/SuGaR)
- [ ] [Neuralangelo: High-Fidelity Neural Surface Reconstruction.](https://github.com/NVlabs/neuralangelo) not support windows
- [ ] [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl) (NeuS hash version) not support windows
- [ ] [Meshing Neural SDFs for Real-Time View Synthesis](https://github.com/hugoycj/torch-bakedsdf)

## DTU dataset
You can download the preprocessed data from [here](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9).
You also need to download the ground truth [DTU point cloud](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9).
After downloading the dataset, you should organize your data like this:
```
datasets
|-- DTU_mask  # preprocessed data
    |-- scan122
    ...
|-- DTU  # official data
    |-- ObsMask
    |-- Points
...
```
Recommend that one view for every eight is taken as the test dataset.

Most of methods have provide the evaluation scripts for dtu datasets. You can follow the guidance in its README.md to compute the metrics.

If the method doesn't provide evaluation scripts, yYou can calculate the metrics for each case separately and calculate its average according to the guidelines below. 

## Evaluation Metrics
0. Run specific method and *tuning hyperparameters*  (set voxel_size=0.5 if possible)
tips：
- If you encounter some problems in the installation, please google it first. Most of the issues have a solution in github issues.
- It is recommended to install the repository on the Windows system, most methods use Open3d, which is not feasible in the Liunx system of the Shanghai science and technology cluster.
- If you are unable to compile a submodule via pip and get an error like ```CUDA runtime not found```. This could be due to a mismatch between your Nivdia Diver, cuda toolkits, and c++ CUDA_HOME, please check these items. 
- If you get the error ```CalledProcessError: Command '['ninja', '-v']```, change the line ```cmdclass={'build_ext': BuildExtension}``` in the setup.py to ```cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}```.
- If you get the error ```RuntimeError: Error compiling objects for extension```, follow these step : 1. check the path of ninja using `where ninja
`. 2. delete the path in the System variable Path. 
- If you get the error ```error: command 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin\\nvcc.exe' failed with exit code 2```, install Microsoft Visual Studio MSVC following [this](https://blog.csdn.net/qq_21488193/article/details/134924533) or refer https://github.com/facebookresearch/detectron2/issues/2390#issuecomment-746335999
- When git clone is too slow or error: RPC fails, run ```git clone http://github.com/large-repository --depth 1```  ```cd large-repository``` ```git fetch --unshallow```.

- If you get the error ```ModuleNotFoundError: No module named 'diff_gaussian_rasterization' or 'simple-knn'```, run  ```cd submodules\diff_gaussian_rasterization or simple-knn```  ```python setup.py install```.
- If you failed ```pip install pytorch3d```, run  ```pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.6+pt<version of pytorch>cu<version of CUDA>```.
- If you failed ```pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch```, you can clone the repository and build locally.
- If you get ```MemroyError``` in ```multiprocessing```, reduce `num_workers` in your code.
- If you get ```"Distributed package doesn't have NCCL " ```, switch backend setting from `nccl` with `gloo` in your code. `nccl` doesn't sypport for windows. `os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"`
- If you get ```C:\Program Files\Microsoft Visual Studio\2022\Community>pushd C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\ 系统找不到指定的路径。```, run `setup_env.bat`
- If ```pip install -e .``` doesn't work , run `python setup.py install`

### Evaluation for rendered images 
```
python eval_nvs.py --gt <Path to ground truth> --pr <Path to rendered images>  --name <Case Name> --num_images <Number of evaluated images>
```
For example
```
python eval_nvs.py --gt D:\wyh\eval_mvs\dtu122_2dgs\test\ours_30000\gt --pr D:\wyh\eval_mvs\dtu122_2dgs\test\ours_30000\renders  --name DTU_bird --num_images 8

```
Before running teh command, please read these items:
1. GT images are named as format: 00000.png, 00001.png 00002.png ... 
2. The result will be saved in ```./logs/metrics/nvs.log```
3. Both GT images and predicted images will be preprocessed and saved in preprocessed folder in <Path to rendered images>
4. Download SAM checkpoint [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and mode to ```./ckpts```

### Evaluation for mesh
Before evaluating the mesh, please postprocess it for fairness.
```
python postprocess_mesh.py --input <Path to predicted mesh> --num_cluster 1
python postprocess_mesh.py --input D:\wyh\SuGaR\output\refined_mesh\scan122\sugarfine_3Dgs7000_sdfestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1_postprocessed.obj --num_cluster 1
```

```
python eval_mesh.py --pr_mesh <Path to predicted mesh directory> --pr_type "mesh" or "pcd"   --gt_mesh <Path to ground truth directory> --gt_mesh_colmap <Path to ground truth colmap directory> --gt_mesh_mask <Path to ground truth mesh> --downsample
```
For example
```
python eval_mesh.py --pr_mesh D:\wyh\eval_mvs\dtu122\train\ours_30000\fuse_post.ply --pr_dir D:\wyh\eval_mvs\dtu122 --pr_type mesh --gt_mesh D:\wyh\eval_mvs\dtu_data\MVS_dataset --gt_mesh_colmap D:\wyh\eval_mvs\dtu_data --gt_mesh_mask D:\wyh\eval_mvs\dtu_data\mask\scan122.ply
```
- If you want to downsample meshes, add ```--downsample``` at the end of command. 
- ```--threshold``` is the threshold for computing F-score. Recommend tuning it based on dis_r and dis_p 

Before running teh command, please read these items:
1. The result will be printed in terminal, please record them including *chamchamfer*, *iou* and  *f score* 
2. The preprocessed mesh will be saved in preprocessed folder in ``` <Path to predicted mesh> \eval_dtu```


## Reporting Results
Present the evaluation results in a clear and concise manner. The visualization results and tables should be illustrated in [slides](./res.pptx). 
The slides should include:
1. Method name
2. Hyperparameters of method (iterations, training dataset size, tese dataset size)
3. Visualization rendered color results, normal map  (GT and rendered)
4. Visualization of mesh (if the format can be inserted in slides)

## Conclusion
Please communicate in a timely manner

## Related projects
We collect code from following projects. We thanks for the contributions from the open-source community!
[SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer/tree/main)

[DTUeval-python](https://github.com/jzhangbs/DTUeval-python)

[TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master)
