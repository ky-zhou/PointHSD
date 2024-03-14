# Joint Learning for Scattered Point Cloud Understanding with Hierarchical Self-Distillation


## Datasets

ModelNet40: download ```modelnet40_normal_resampled``` from the official site for joint learning and the processed h5 version from PointNet official for classification-only (downstream) tasks.

ScanOjbectNN: download the h5 file from the official site. Use the variant: PB_T50_RS.

ShapeNetpart: download ```shapenet_part_seg_hdf5_data``` from the official site.

Data structure for preprocessing should be like:
```
|-datasets/
|--modelnet40_normal_resampled/
|--modelnet40_ply_hdf5_2048/
|--h5_files/
|--shapenet_part_seg_hdf5_data/
```

Prepare the data using ```ModelNetDataLoader```, ```ScanOjbectProcesser``` or ```ShapeNetPartV0Processer``` in ```utils/preprocess.py```.

Data structure for training and testing should be like:
```
-datasets
|--ModelNet40/
|---test/
|---train/
|---modelnet40_shape_names.txt
|---modelnet40_test.txt
|---modelnet40_train.txt
|--ScanObjectNN*/
|---test/
|---train/
|---scanobjectnn_shape_names.txt
|---scanobjectnn_test.txt
|---scanobjectnn_train.txt
|--ShapeNetPartV0/
|---test/
|---train/
|--shapenet_part_seg_hdf5_data/
|---synsetoffset2category.txt
|---*_hdf5_file_list.txt
```
where *** are file or folder names, and * should be ```_75_s*_k*``` for without background and ```_nobg_75_s*_k*``` for without background.

## Environment Configuration

#### Install Python Denpendencies

```shell
cd PointHSD
pip install -r requirements.txt
```

#### Build PyTorch Extensions

**NOTE:** PyTorch >= 1.4 of cuda version are required.

```shell
cd pointnet2_ops_lib
python setup.py install

cd ..

cd Chamfer3D
python setup.py install
```

You need to update the file path of the datasets:

```shell
__C.DATASETS.MODELNET.PARTIAL_POINTS_PATH        = './datasets/ModelNet40/%s/partial/%s/%s/%s-%d.pcd'
__C.DATASETS.MODELNET.COMPLETE_POINTS_PATH       = './datasets/ModelNet40/%s/complete/%s/%s.pcd'
__C.DATASETS.SCANOBNN.TYPE                       = 'ScanObjectNN_nobg_75_s64_k16'

# Dataset Options: ModelNet40, ScanObjectNN, ModelNet10, ShapeNetPart
__C.DATASET.TRAIN_DATASET                        = 'ScanObjectNN'
__C.DATASET.TEST_DATASET                         = 'ScanObjectNN'
```


## Getting Started

We have the following models indexed as:

### Classification only
17: only downstream + HSD
18: only downstream
### Joint Learning for classification
12: baselines (upstream+downstream)
21: HSD (upstream+downstream)
### Storing data for Information Plane
25: HSD that can store data for calculating and plotting Informaton plane 
### Joint Learning for part segmentation
9: HSD for part segmentation
10: baselines for part segmentation

To train PointHSD, you can simply use the following command:

```shell
python main*.py
```

To test or inference, you should specify the path of checkpoint if the config_pcn.py file. Specifically, we have provided test data checkpoints for J+PointNet++-HSD on ModelNet40 32*16, and J+PointNet++-HSD on ShapeNetPart.
```shell
__C.CONST.WEIGHTS                                = "path to your checkpoint" # ./output/2023-12-09T21:08:45.179615/best_model_xxxx.pth'
```

then use the following command:

```shell
python main*.py --test
python main*.py --inference
```

Note that partial input, ground truth, and predicted points can be saved by uncommenting the "save data" section in ```test_*.py```.

## Information Plane Plotting

Use ```IDNNs/main2.py``` to plot the information plane.

## License

This project is open sourced under MIT license.

## Cite the Related Papers

```
@inproceedings{zhou2022zero,
  title={Cascaded Network with Hierarchical Self-Distillation for Sparse Point Cloud Classification},
  author={Zhou, Kaiyue and Dong, Ming and Arslanturk, Suzan},
  booktitle={2024 IEEE International Conference on Multimedia and Expo (ICME)},
  year={2024},
  organization={IEEE}
}
@article{zhou2023joint,
  title={Joint Learning for Scattered Point Cloud Understanding with Hierarchical Self-Distillation},
  author={Zhou, Kaiyue and Dong, Ming and Zhi, Peiyuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:2312.16902},
  year={2023}
}
```
