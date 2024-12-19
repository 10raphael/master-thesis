# Master project "Consistent semantic mapping for LiDAR-visual and RGB-D data"

## General set-up
Use your default package manager or build from source:
- cmake > 3.5
- PCL > 1.2
- OpenCV

Additionally, install [libpointmatcher](https://github.com/norlab-ulaval/libpointmatcher?tab=readme-ov-file#quick-start)

then:
```bash
mkdir build && cd build
cmake ..
make
```

## Label propagation on ScanNet using predictions from EMSANet

1. Request access to the [ScanNet](https://github.com/ScanNet/ScanNet?tab=readme-ov-file#scannet-data) dataset and use the script provided in email to download the sequence
3. Use this provided [script](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python#data-exporter), to extract the data needed from the .sens file
4. To get the ground thruth semantic masks, you will need to run [this](https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/2d_helpers/convert_scannet_label_image.py) in order for the labels to be in the correct taxonomy. They are provided in scannet200, and we will use nyu40id.
5. In order to generate the EMSANet predictions, [follow this](https://github.com/TUI-NICR/panoptic-mapping?tab=readme-ov-file#usage) and generate the predictions for the sequence you downloaded. As we don't want the run the inference over the whole dataset, insert a list ```scenes = ['scene0xxx_xx', ...]``` [here](https://github.com/TUI-NICR/nicr-scene-analysis-datasets/blob/af4c3563e6eaecbfb659a8bf735608a46c32c0e7/nicr_scene_analysis_datasets/datasets/scannet/prepare_dataset.py#L585)
6. After that, adjust the paths on l. 102 and l.104 of ```lprop_scannet.cpp```. 
- ```dataset_path``` is the path to what the folder containing the folders ```depth```, ```pose``` and ```intrinsic```, that was generated in step 2.
- ```mask_in``` can be either the masks in the correct taxonomy from step 4, or the predicted masks from step 5
7. ```make ./lprop_scannet```, then ```./lprop_scannet num_frames```
This will generate a .pcd, which contains the resulting map

## Label propagation on semanticKITTI using Rangenet predictions (and the confidence scores)
1. Download the [SemanticKITTI](https://www.semantic-kitti.org/dataset.html#download) dataset, and store all components in the ```.../dataset/sequences/...``` of the sequence to be run. 
2. Download the [predictions](https://github.com/PRBonn/lidar-bonnetal?tab=readme-ov-file#predictions-from-models) and move the ```prediction``` folder to the same directory as the sequence from step 1.
3. In case confidence scores are used, append [this](https://github.com/PRBonn/lidar-bonnetal/blob/master/train/tasks/semantic/modules/user.py) to store the confidence scores. 
   These are the changes to be made:

   line 123f:
   ```bash
   conf_score = proj_output[0].max(dim=0).values
   topk_values, topk_indices = proj_output[0].topk(5, dim=0)
   ```

   line 134f:
   ```bash
   unproj_conf = conf_score[p_y, p_x]
   unproj_argmax_topk = topk_indices[:, p_y, p_x]
   unproj_conf_topk = topk_values[:, p_y, p_x]
   ```
   line 147f:
   ```bash
   # conf scores
   conf_np = unproj_conf.detach().cpu().numpy()
   conf_np = conf_np.reshape((-1)).astype(np.float32)
   ```
   line 155f:
   ```bash
   conf_path = path + "c"
   conf_np.tofile(conf_path)
   ```
This will generate a file xxxxx.label and xxxxx.labelc. Store these all in the same folder for the sequence.

5. ```make ./lprop_semkitti```, then ```./lprop_semkitti num_frames```
This will generate a .pcd, which contains the resulting map.

## Miscellaneous
The code runs on Apple M3 Max, the packages where install using homebrew, the code was developped on vscode. 
