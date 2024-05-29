# Evaluation of Multi-View Stereo (MVS) Methods

## Introduction
This document provides instructions for evaluating Multi-View Stereo (MVS) methods. MVS is a computer vision technique that reconstructs a 3D scene from multiple 2D images taken from different viewpoints.
## Methods to evaluate
- [x] [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://github.com/hbb1/2d-gaussian-splatting)
- [ ] [SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering](https://github.com/Anttwo/SuGaR)
- [ ] [Neuralangelo: High-Fidelity Neural Surface Reconstruction.](https://github.com/NVlabs/neuralangelo)
- [ ] [Meshing Neural SDFs for Real-Time View Synthesis.](https://github.com/hugoycj/torch-bakedsdf)
- [ ] [SUNDAE: Spectrally Pruned Gaussian Fields with Neural Compensation.](https://github.com/RunyiYang/SUNDAE)
## Baseline (Mesh)
- [x] [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction.](https://github.com/Anttwo/SuGaR)
- [ ] [Volume Rendering of Neural Implicit Surfaces.](https://github.com/lioryariv/volsdf)
## Dataset
```
<case_name>
|-- images             
    |-- 00000.png        # target image for each view
    |-- 00001.png
    ...
|-- mesh
    |-- materials      
    |-- meshes         # target meshes
    ...
...
```
The total number of images is 150. The elevations of view are 0°，45° and 23°. Each elevation angle corresponds to 50 azimuths, evenly distributed. Recommend using 132 images for training and 18 images for testing (One view for every eight is taken as the test dataset). The camera parameters is recorded in [cameras.json](./cameras.json).

## Evaluation Metrics
0. Run specific method and *tuning hyperparameters*  (set voxel_size=0.5 if possible)
### Evaluation for rendered images 
```
python eval_nvs.py --gt <Path to ground truth> --pr <Path to rendered images>  --name <Case Name> --num_images <Number of evaluated images>
# eg.
python eval_nvs.py --gt D:\2d-gaussian-splatting\output\ec536168-0\test\ours_30000\gt --pr D:\2d-gaussian-splatting\output\ec536168-0\test\ours_30000\renders  --name 2DGS_LEGO --num_images 18

```
Before running teh command, please read these items:
1. GT images are named as format: 00000.png, 00001.png 00002.png ... 
2. The result will be saved in ```./logs/metrics/nvs.log```
3. Both GT images and predicted images will be preprocessed and saved in preprocessed folder in <Path to rendered images>
4. Download SAM checkpoint [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and mode to ```./ckpts```

### Evaluation for mesh
This version need to be tested. Please wait for updating.
```
python eval_mesh.py --pr_mesh <Path to predicted mesh> --gt_dir <Path to ground truth mesh directory> --gt_mesh <Path to ground truth mesh> --name <Case name>

# eg.
python eval_mesh.py --pr_mesh D:\2d-gaussian-splatting\output\ec536168-0\train\ours_30000\fuse_unbounded_post.ply --gt_dir D:\2d-gaussian-splatting\output\ec536168-0\train\ours_30000 --gt_mesh D:\Free3D\MVS_data\render_res\LEGO_Duplo_Build_and_Play_Box_4629\mesh\meshes\model.obj --name LEGO_Duplo_Build_and_Play_Box_4629

```
Before running teh command, please read these items:
1. The format of mesh can be ```.obj ```, ```.ply ```...
2. The result will be saved in ```./logs/metrics/mesh.log```
3. The preprocessed mesh will be saved in preprocessed folder in ```./logs```


## Reporting Results
Present the evaluation results in a clear and concise manner. The visualization results and tables should be illustrated in [slides](./res.pptx). 
The slides should include:
1. Method name
2. Hyperparameters of method (iterations, training dataset size, tese dataset size)
3. Visualization rendered color results, normal map  (GT and rendered)
4. Visualization of mesh (if the format can be inserted in slides)

## Conclusion
Please communicate in a timely manner

