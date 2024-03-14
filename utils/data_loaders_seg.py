# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:21:32
# @Email:  cshzxie@gmail.com

import json
import logging
import numpy as np
import random
import torch.utils.data.dataset
import open3d as o3d
import utils.data_transforms
from enum import Enum, unique
from tqdm import tqdm
from utils.io import IO

# label_mapping_partv0 = {
#     3: '03001627',
#     6: '04379243',
#     5: '04256520',
#     1: '02933112',
#     4: '03636649',
#     2: '02958343',
#     0: '02691156',
#     7: '04530566'
# }
label_mapping = {
    '03001627': 3,
    '04379243': 6,
    '04256520': 5,
    '02933112': 1,
    '03636649': 4,
    '02958343': 2,
    '02691156': 0,
    '04530566': 7
}
label_mapping_partv0 = {
    '02691156': 0,  #Airplane
    '02773838': 1,  #Bag
    '02954340': 2,  #Cap
    '02958343': 3,  #Car
    '03001627': 4,  #Chair
    '03261776': 5,  #Earphone
    '03467517': 6,  #Guitar
    '03624134': 7,  #Knife
    '03636649': 8,  #Lamp
    '03642806': 9,  #Laptop
    '03790512': 10, #Motorbike
    '03797390': 11, #Mug
    '03948459': 12, #Pistol
    '04099429': 13, #Rocket
    '04225987': 14, #Skateboard
    '04379243': 15, #Table
}
label_mapping2 = {
    'airplane' : 0,
    'bathtub' : 1,
    'bed' : 2,
    'bench' : 3,
    'bookshelf' : 4,
    'bottle' : 5,
    'bowl' : 6,
    'car' : 7,
    'chair' : 8,
    'cone' : 9,
    'cup' : 10,
    'curtain' : 11,
    'desk' : 12,
    'door' : 13,
    'dresser' : 14,
    'flower_pot' : 15,
    'glass_box' : 16,
    'guitar' : 17,
    'keyboard' : 18,
    'lamp' : 19,
    'laptop' : 20,
    'mantel' : 21,
    'monitor' : 22,
    'night_stand' : 23,
    'person' : 24,
    'piano' : 25,
    'plant' : 26,
    'radio' : 27,
    'range_hood' : 28,
    'sink' : 29,
    'sofa' : 30,
    'stairs' : 31,
    'stool' : 32,
    'table' : 33,
    'tent' : 34,
    'toilet' : 35,
    'tv_stand' : 36,
    'vase' : 37,
    'wardrobe' : 38,
    'xbox' : 39,
}
@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch): # allocate batch size of data, and output one more dimension: labels
    taxonomy_ids = [] # will be size of B
    model_ids = []
    data = {}
    labels = [] # will be size of B

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        labels.append(label_mapping[sample[0]])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items(): #k: partial_cloud or gtcloud
        data[k] = torch.stack(v, 0) #B, 2048 or 16384, 3

    return taxonomy_ids, model_ids, data, labels

def collate_fn3(batch): # allocate batch size of data, and output one more dimension: labels
    taxonomy_ids = [] # will be size of B
    model_ids = []
    data = {}
    data_label = {} # will be size of B
    data_class_label = []

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        data_class_label.append(label_mapping[sample[0]])
        _data = sample[2]
        _data_label = sample[3]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
        for k, v in _data_label.items():
            if k not in data_label:
                data_label[k] = []
            data_label[k].append(v)

    # labels = torch.stack(labels, 0)
    for k, v in data.items(): #k: partial_cloud or gtcloud
        data[k] = torch.stack(v, 0) #B, 2048 or 16384, 3
    #     print(f'v1 v: {v[0].shape, type(v)}')
    #     print(f'v1 data[k]: {data[k].shape}')
    # print(f'v1 data[gtcloud]: ', data['gtcloud'].shape)
    for k, v in data_label.items(): #k: partial_cloud or gtcloud
        # print(f'v2: {v[0].shape, type(v)}')
        data_label[k] = torch.stack(v, 0) #B, 2048 or 16384, 3
    # print(f'v1 data_label[gtlabel]: ', data_label['gtlabel'].shape)

    return taxonomy_ids, model_ids, data, data_label, data_class_label

def collate_fn4(batch): # allocate batch size of data, and output one more dimension: labels
    taxonomy_ids = [] # will be size of B
    model_ids = []
    data = {}
    data_label = {} # will be size of B
    data_class_label = []

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        data_class_label.append(label_mapping_partv0[sample[0]])
        _data = sample[2]
        _data_label = sample[3]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
        for k, v in _data_label.items():
            if k not in data_label:
                data_label[k] = []
            data_label[k].append(v)

    data_class_label = torch.tensor(data_class_label)
    # print(f'data_class_label {data_class_label}')
    for k, v in data.items(): #k: partial_cloud or gtcloud
        data[k] = torch.stack(v, 0) #B, 2048 or 16384, 3
    #     print(f'v1 v: {v[0].shape, type(v)}')
    #     print(f'v1 data[k]: {data[k].shape}')
    # print(f'v1 data[gtcloud]: ', data['gtcloud'].shape)
    for k, v in data_label.items(): #k: partial_cloud or gtcloud
        # print(f'v2: {v[0].shape, type(v)}')
        data_label[k] = torch.stack(v, 0) #B, 2048 or 16384, 3
    # print(f'v1 data_label[gtlabel]: ', data_label['gtlabel'].shape)

    return taxonomy_ids, model_ids, data, data_label, data_class_label

code_mapping = {
    'plane': '02691156',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'couch': '04256520',
    'table': '04379243',
    'watercraft': '04530566',
}

def read_ply(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data, data_label = {}, {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            # print(file_path)
            data[ri] = IO.get(file_path).astype(np.float32)
            # print(f'data[ri]: {data[ri].shape}')
        for ri in self.options['required_labels']:
            label_path = sample['%s_path' % ri]
            data_label[ri] = IO.get(label_path).astype(np.float32)
            # print(f'label: {data_label[ri].shape}')
        # print(2)
        if self.transforms is not None:
            data = self.transforms(data) # call the objects (e.g., 'objects': ['partial_cloud', 'gtcloud']) iteratively
            data_label = self.transforms(data_label)
        # print(f'Dataset data gt:  ', data['gtcloud'].shape)
        # print(f'Dataset data gtlabel:  ', data_label['gtlabel'].shape)

        return sample['taxonomy_id'], sample['taxonomy_name'], data, data_label


class ShapeNetPartV0DataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.SEGMENTATION_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        # print(f'ShapeNetDataLoader: get_dataset')
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'required_labels': ['gtlabel'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RescaleSegPoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud', 'gtcloud']
            },
            #     {
            #     'callback': 'RescaleSegPoints',
            #     'parameters': {
            #         'n_points': cfg.DATASETS.SHAPENET.N_POINTS
            #     },
            #     'objects': ['gtcloud']
            # },
                {
                'callback': 'RescaleSegLabels',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['gtlabel']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud', 'gtlabel']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'RescaleSegPoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'RescaleSegLabels',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['gtlabel']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud', 'gtlabel']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]
            for s in samples:
                label_path = cfg.DATASETS.SHAPENET.COMPLETE_LABELS_PATH % (subset, dc['taxonomy_id'], s)
                if subset == 'test':
                    gt_path = cfg.DATASETS.SHAPENET.SEG_COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    file_list.append({'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'taxonomy_name': dc['taxonomy_name'],
                    'gtlabel_path': label_path,
                    'partial_cloud_path': gt_path.replace('complete', 'partial'),
                    'gtcloud_path': gt_path})
                else:
                    file_list.append({
                    'taxonomy_id':  dc['taxonomy_id'],
                    'model_id': s,
                    'taxonomy_name': dc['taxonomy_name'],
                    'gtlabel_path': label_path,
                    'partial_cloud_path': [
                        cfg.DATASETS.SHAPENET.SEG_PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s, s, i)
                        for i in range(n_renderings)
                    ],
                    'gtcloud_path':
                        cfg.DATASETS.SHAPENET.SEG_COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                })

        # print(f'_get_file_list file_list {len(file_list)}')

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class ShapeNetDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        # print(f'ShapeNetDataLoader: get_dataset')
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'required_labels': ['gtlabel'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RescaleSegPoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud', 'gtcloud']
            },
            #     {
            #     'callback': 'RescaleSegPoints',
            #     'parameters': {
            #         'n_points': cfg.DATASETS.SHAPENET.N_POINTS
            #     },
            #     'objects': ['gtcloud']
            # },
                {
                'callback': 'RescaleSegLabels',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['gtlabel']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud', 'gtlabel']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'RescaleSegPoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RescaleSegLabels',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['gtlabel']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud', 'gtlabel']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []
        with open(cfg.DATASETS.SHAPENET.SEGMENTATION_FILE_PATH % subset) as f:
            shapenet_samples = json.loads(f.read())
            print(f'shapenet_samples: {shapenet_samples}')

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]
            for s in samples:
                target_sample_in_shapedata = 'shape_data/%s/%s' % (dc['taxonomy_id'], s)
                # category_name, s = dc.split('/')[-2], dc.split('/')[-1]
                label_path = cfg.DATASETS.SHAPENET.COMPLETE_LABELS_PATH % (subset, dc['taxonomy_id'], s)
                if subset == 'test':
                    if target_sample_in_shapedata in shapenet_samples:
                        gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                        file_list.append({'taxonomy_id': dc['taxonomy_id'],
                        'model_id': s,
                        'taxonomy_name': dc['taxonomy_name'].capitalize (),
                        'gtlabel_path': label_path,
                        'partial_cloud_path': gt_path.replace('complete', 'partial'),
                        'gtcloud_path': gt_path})
                else:
                    if target_sample_in_shapedata in shapenet_samples:
                        file_list.append({
                        'taxonomy_id':
                            dc['taxonomy_id'],
                        'model_id': s,
                        'taxonomy_name': dc['taxonomy_name'].capitalize (),
                        'gtlabel_path': label_path,
                        'partial_cloud_path': [
                            cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s, i)
                            for i in range(n_renderings)
                        ],
                        'gtcloud_path':
                            cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                    })

        # print(f'_get_file_list file_list {len(file_list)}')

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class ShapeNetCarsDataLoader(ShapeNetDataLoader):
    def __init__(self, cfg):
        super(ShapeNetCarsDataLoader, self).__init__(cfg)

        # Remove other categories except cars
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']


class ModelNet40DataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        # self.dataset_categories = []
        with open(cfg.DATASETS.MODELNET.CATEGORY_FILE_PATH) as f:
            # print(f'ModelNet40DataLoader: {f}')
            self.dataset_categories = np.loadtxt(f, dtype=str)

    def get_dataset(self, subset):
        # print(f'ShapeNetDataLoader: get_dataset')
        n_renderings = self.cfg.DATASETS.MODELNET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        # print(0)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RescalePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.MODELNET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'RescalePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.MODELNET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for taxonomy_id in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [Name=%s]' % taxonomy_id)
            samples = np.loadtxt('./datasets/ModelNet/modelnet40_%s.txt' % subset, dtype=str)
            # samples = dc[subset]

            for s in tqdm(samples, leave=False):
                category_name = s.split('_')[0]
                # if len(info) == 2:
                #     category_name = info[0]
                # else:
                #     category_name = '_'.join(info[:-1])
                if subset == 'test':
                    if category_name in taxonomy_id:
                        gt_path = cfg.DATASETS.MODELNET.COMPLETE_POINTS_PATH % (subset, taxonomy_id, s)
                        file_list.append({'taxonomy_id': taxonomy_id,
                        'partial_cloud_path': gt_path.replace('complete', 'partial'),
                        'gtcloud_path': gt_path})
                else:
                    if category_name in taxonomy_id:
                        file_list.append({
                            'taxonomy_id': taxonomy_id,
                            'partial_cloud_path': [
                                cfg.DATASETS.MODELNET.PARTIAL_POINTS_PATH % (subset, taxonomy_id, s, s, i)
                                for i in range(n_renderings)
                            ],
                            'gtcloud_path':
                                cfg.DATASETS.MODELNET.COMPLETE_POINTS_PATH % (subset, taxonomy_id, s),
                        })

        # print(f'_get_file_list file_list {len(file_list)}')

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class ModelNet10DataLoader(ModelNet40DataLoader):
    def __init__(self, cfg):
        super(ModelNet10DataLoader, self).__init__(cfg)

        # Remove other categories except ModelNet10
        with open(cfg.DATASETS.MODELNET.CATEGORY_FILE_PATH.replace('40', '10')) as f:
            modelnet10_categories = np.loadtxt(f, dtype=str)
        self.dataset_categories = [dc for dc in self.dataset_categories if dc in modelnet10_categories]


class Completion3DDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud'] if subset == DatasetSubset.TEST else ['partial_cloud', 'gtcloud']

        return Dataset({
            'required_items': required_items,
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                   'n_points': cfg.CONST.N_INPUT_POINTS
               },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            },{
                    'callback': 'ScalePoints',
                    'parameters': {
                        'scale': 0.85
                    },
                    'objects': ['partial_cloud', 'gtcloud']
                },
                {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        elif subset == DatasetSubset.VAL:
            return utils.data_transforms.Compose([{
                'callback': 'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            },{
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                    'gtcloud_path':
                    cfg.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class Completion3DPCCTDataLoader(Completion3DDataLoader):
    """
    Dataset Completion3D containing only plane, car, chair, table
    """
    def __init__(self, cfg):
        super(Completion3DPCCTDataLoader, self).__init__(cfg)

        # Remove other categories except couch, chairs, car, lamps
        cat_set = {'02691156', '03001627', '02958343', '04379243'} # plane, chair, car, table
        # cat_set = {'04256520', '03001627', '02958343', '03636649'}
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] in cat_set]


class KittiDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.KITTI.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud', 'bounding_box']

        return Dataset({'required_items': required_items, 'shuffle': False}, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        return utils.data_transforms.Compose([{
            'callback': 'NormalizeObjectPose',
            'parameters': {
                'input_keys': {
                    'ptcloud': 'partial_cloud',
                    'bbox': 'bounding_box'
                }
            },
            'objects': ['partial_cloud', 'bounding_box']
        }, {
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
            },
            'objects': ['partial_cloud']
        }, {
            'callback': 'ToTensor',
            'objects': ['partial_cloud', 'bounding_box']
        }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': cfg.DATASETS.KITTI.PARTIAL_POINTS_PATH % s,
                    'bounding_box_path': cfg.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH % s,
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'Completion3D': Completion3DDataLoader,
    'Completion3DPCCT': Completion3DPCCTDataLoader,
    'ShapeNet': ShapeNetDataLoader,
    'ShapeNetCars': ShapeNetCarsDataLoader,
    'KITTI': KittiDataLoader,
    'ModelNet40': ModelNet40DataLoader,
    'ModelNet10': ModelNet10DataLoader,
    'ShapeNetSeg': ShapeNetDataLoader,
    'ShapeNetPart': ShapeNetPartV0DataLoader,
}  # yapf: disable

