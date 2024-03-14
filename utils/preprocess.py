import json
import logging
import numpy as np
import random
import torch.utils.data.dataset
import open3d as o3d
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
import utils.data_transforms
from enum import Enum, unique
from tqdm import tqdm
from utils.io import IO
from config_pcn import cfg
import collections
import h5py


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

@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


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
        data = {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            # print(file_path)
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)
        # print(f'Dataset: ', sample['label'])

        return sample['taxonomy_id'], sample['model_id'], data#, sample['label']


class ShapeNetDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            # print(f'ShapeNetDataLoader: {f}')
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        # print(f'ShapeNetDataLoader: get_dataset')
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
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
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
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

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                if subset == 'test':
                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    file_list.append({'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    # 'label': dc['taxonomy_label'],
                    'partial_cloud_path': gt_path.replace('complete', 'partial'),
                    'gtcloud_path': gt_path})
                else:
                    file_list.append({
                        'taxonomy_id':
                            dc['taxonomy_id'],
                        'model_id':
                            s,
                        # 'label': dc['taxonomy_label'],
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


class ShapeNetPartDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open('../datasets/shape_data/train_test_split/shuffled_val_file_list.json') as f:
            self.dataset_categories = json.loads(f.read())
            # print(f'ShapeNetPartDataLoader: {self.dataset_categories}')
        pcd = o3d.io.read_point_cloud("/home/kz/Documents/SnowflakeNet/datasets/PCN/train/partial/02933112/1a9a91aa5e3306ec5938fc2058ab2dbe/05.pcd")
        out_arr = np.asarray(pcd.points)
        out2 = np.loadtxt('/home/kz/Documents/SnowflakeNet/datasets/shape_data/03642806/2ce3a50ca6087f30d8e007cc6755cce9.txt') # different size
        print(f'out_arr {out_arr.shape}')
        print(f'out2 {out2.shape}')
        pcn_path = '../datasets/PCN/%s/complete/%s/%s.pcd'
        snpart_path = '../datasets/shape_data/%s/%s.pcd'
        split_set = ['train', 'test', 'val']
        snpart_set_path = '../datasets/shape_data/train_test_split/shuffled_%s_file_list.json'

    def get_dataset(self, subset):
        # print(f'ShapeNetDataLoader: get_dataset')
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
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
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
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

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                if subset == 'test':
                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    file_list.append({'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    # 'label': dc['taxonomy_label'],
                    'partial_cloud_path': gt_path.replace('complete', 'partial'),
                    'gtcloud_path': gt_path})
                else:
                    file_list.append({
                        'taxonomy_id':
                            dc['taxonomy_id'],
                        'model_id':
                            s,
                        # 'label': dc['taxonomy_label'],
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


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm, skip_initial=False, indices_dtype=np.int32, distances_dtype=np.float32):
    """Batch operation of farthest point sampling
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    # xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = np.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = np.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = np.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = np.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = np.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = np.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = np.minimum(min_distances, dist)
    return indices, distances


class ModelNetDataLoader(object):
    def __init__(self, cfg, gen_complete=False, gen_partial=False, num_seed=64, k=16, modes=('test', 'train')):
        self.cfg = cfg
        for mode in modes:
            print('Generating mode =', mode)
            pcd_path = '../datasets/modelnet40_normal_resampled/%s/%s.txt'
            pcd_list = np.loadtxt('../datasets/modelnet40_normal_resampled/modelnet40_%s.txt' % mode, dtype=str)

            for pcd_name in tqdm(pcd_list):
                info = pcd_name.split('_')
                if len(info) == 2:
                    category_name = info[0]
                else:
                    category_name = '_'.join(info[:-1])
                cur_pcd = np.loadtxt(pcd_path % (category_name, pcd_name), delimiter=',')[:, :3]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cur_pcd)
                # 1k or 10k points to be the ground truth, comment or uncomment the statement below
                pcd = pcd.farthest_point_down_sample(1024)        
                if gen_complete:
                    target_complete_path = '../datasets/ModelNet40/%s/complete/%s/' % (mode, category_name)
                    if not os.path.exists(target_complete_path):
                        os.makedirs(target_complete_path)
                    tartet_path = os.path.join(target_complete_path, '%s.pcd' % pcd_name)
                    o3d.io.write_point_cloud(tartet_path, pcd)

                if gen_partial:
                    cur_pcd = np.asarray(pcd.points) # GT with FPS
                    # print(type(cur_pcd))
                    if mode == 'train':
                        target_partial_path = '../datasets/ModelNet40/%s/partial64*4/%s/%s/' % (mode, category_name, pcd_name)
                        num_views = 8
                    else:
                        target_partial_path = '../datasets/ModelNet40/%s/partial64*4/%s/' % (mode, category_name)
                        num_views = 1
                    if not os.path.exists(target_partial_path):
                        os.makedirs(target_partial_path)
                    # each view contains 10 unmasked seeds
                    for view_idx in range(num_views):
                        # randomly select the seeds
                        rand_idx = np.random.choice(len(cur_pcd), num_seed) # number of seeds (views)
                        seeds = cur_pcd[rand_idx]  # need 8 seeds for training set, 1 for testing set
                        tree = KDTree(cur_pcd, leaf_size=2)
                        dist, ball_idx = tree.query(seeds, k=k)  # each view contains num_seed * 400 points queried
                        masked_pcd = []
                        for seed_idx in range(len(ball_idx)):
                            masked_patch = cur_pcd[ball_idx[seed_idx]]
                            masked_pcd.append(masked_patch)
                        masked_pcd = np.concatenate(masked_pcd, 0)
                        # print(f'unmasked_pcd: {unmasked_pcd.shape}')
                        # np.savetxt('../test%s.xyz'%pcd_name, unmasked_pcd) # print for visualization
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(masked_pcd)
                        target_name = pcd_name+'-%d.pcd'%view_idx if mode=='train' else pcd_name+'.pcd'
                        o3d.io.write_point_cloud(os.path.join(target_partial_path, target_name), pcd)


class ShapeNetProcesser(object): # extract common files from PCN shapenet and original shapenet
    def __init__(self, cfg, gen_complete=False, gen_partial=False, mode='train'):
        self.cfg = cfg
        # mode = 'train'
        pcd_path = '../datasets/shape_data/%s/%s.txt'
        seg_json = '../datasets/ShapeNetSeg.json'
        original_json = '../datasets/ShapeNet.json'
        """generate new json for segmentation only due to some files not exist"""
        # with open(original_json) as f:
        #     self.dataset_categories = json.loads(f.read())
        # seg_dicts = []
        # for dc in self.dataset_categories:
        #     logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
        #     exist_files_test, exist_files_train = [], []
        #     category_name = dc['taxonomy_id']
        #     print(f'category: {category_name}')
        #     for mode in ('test', 'train'):
        #         samples = dc[mode]
        #
        #         for s in tqdm(samples, leave=False):
        #             file = pcd_path % (category_name, s)
        #             if os.path.isfile(file):
        #                 if mode == 'train':
        #                     exist_files_train.append(s)
        #                 else:
        #                     exist_files_test.append(s)
        #     new_dict = {
        #         'taxonomy_id': dc['taxonomy_id'],
        #         'taxonomy_name': dc['taxonomy_name'],
        #         'test': exist_files_test,
        #         'train': exist_files_train,
        #     }
        #     if exist_files_test or exist_files_train:
        #         seg_dicts.append(new_dict)
        # with open(seg_json, "w") as outfile:
        #     json.dump(seg_dicts, outfile, indent=4)

        """complete"""
        if gen_complete:
            with open(seg_json) as f:
                self.dataset_categories = json.loads(f.read())
            for dc in self.dataset_categories:
                logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
                samples = dc[mode]
                category_name = dc['taxonomy_id']

                for s in tqdm(samples, leave=False):
                    file = pcd_path % (category_name, s)
                    # if os.path.isfile(file):
                    info = np.loadtxt(file, delimiter=' ')[:, :7]
                    cur_pcd, cur_normal, cur_label = info[:, :3], info[:, 3:6], info[:, 6]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cur_pcd)
                    target_complete_path = '../datasets/ShapeNet/%s/complete/%s/' % (mode, category_name)
                    if not os.path.exists(target_complete_path):
                        os.makedirs(target_complete_path)
                    target_path = os.path.join(target_complete_path, '%s.pcd' % s)
                    # target_label_path = os.path.join(target_complete_path, '%s_label.txt' % s)
                    o3d.io.write_point_cloud(target_path, pcd)
                    np.savetxt(os.path.join(target_complete_path, '%s-label.txt' % s), cur_label, fmt='%.5f')

        """partial"""
        if gen_partial:
            with open(seg_json) as f:
                self.dataset_categories = json.loads(f.read())
            whole_pcds = collections.defaultdict(list)

            for dc in self.dataset_categories:
                logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
                samples = dc[mode]
                category_name = dc['taxonomy_id']

                for s in tqdm(samples, leave=False):
                    file = pcd_path % (category_name, s)
                    info = np.loadtxt(file, delimiter=' ')[:, :7]
                    cur_pcd, cur_normal, cur_label = info[:, :3], info[:, 3:6], info[:, 6]
                    whole_pcds[category_name].append(cur_pcd)
                    if mode == 'train':
                        target_partial_path = '../datasets/ShapeNet/%s/partial/%s/%s/' % (mode, category_name, s)
                        num_views = 8
                    else:
                        target_partial_path = '../datasets/ShapeNet/%s/partial/%s/' % (mode, category_name)
                        num_views = 1
                    if not os.path.exists(target_partial_path):
                        os.makedirs(target_partial_path)
                    # each view contains 10 unmasked seeds
                    num_seed = 64
                    for view_idx in range(num_views):
                        # randomly select the seeds
                        rand_idx = np.random.choice(len(cur_pcd), num_seed) # number of seeds (views)
                        seeds = cur_pcd[rand_idx]  # need 8 seeds for training set, 1 for testing set
                        tree = KDTree(cur_pcd, leaf_size=2)
                        dist, ball_idx = tree.query(seeds, k=32)  # each view contains num_seed * 400 points queried
                        masked_pcd, masked_partial_label = [], []
                        for seed_idx in range(len(ball_idx)):
                            masked_patch = cur_pcd[ball_idx[seed_idx]]
                            masked_patch_label = cur_label[ball_idx[seed_idx]]
                            masked_pcd.append(masked_patch)
                            masked_partial_label.append(masked_patch_label)
                        masked_pcd = np.concatenate(masked_pcd, 0)
                        masked_partial_label = np.concatenate(masked_partial_label, 0)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(masked_pcd)
                        target_name = s+'-%d.pcd'%view_idx if mode=='train' else s+'.pcd'
                        o3d.io.write_point_cloud(os.path.join(target_partial_path, target_name), pcd)
                        target_label_path = os.path.join(target_partial_path, '%s-%d-label.txt' % (s, view_idx))
                        np.savetxt(target_label_path, masked_partial_label, fmt='%.5f')


class ShapeNetProcesser2(object): # extract from original shapenet
    def __init__(self, cfg, gen_complete=False, gen_partial=False, mode='train'):
        self.cfg = cfg
        # mode = 'test'
        shapenet_path = '../datasets/'
        target_path = '../datasets/ShapeNet/'
        data_json = '../datasets/shape_data/train_test_split/shuffled_%s_file_list.json' % mode
        # original_json = '../datasets/ShapeNet.json'
        """generate new json for segmentation only due to some files not exist"""
        with open(data_json) as f:
            self.dataset_categories = json.loads(f.read())
        # print(self.dataset_categories)
        seg_dicts = []
        if gen_complete:
            for dc in tqdm(self.dataset_categories, leave=False):
                # logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
                category_name, s = dc.split('/')[-2], dc.split('/')[-1]
                info = np.loadtxt(os.path.join(shapenet_path, dc+'.txt'), delimiter=' ')[:, :7]
                # print(info.shape, category_name)
                cur_pcd, cur_normal, cur_label = info[:, :3], info[:, 3:6], info[:, 6]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cur_pcd)
                # x = pcd.farthest_point_down_sample(2048)
                # x = pcd.random_down_sample(0.9)
                # print(x.dimension)
                # y = np.asarray(x.points)
                target_complete_path = '../datasets/ShapeNetSeg/%s/complete/%s/' % (mode, category_name)
                if not os.path.exists(target_complete_path):
                    os.makedirs(target_complete_path)
                target_path = os.path.join(target_complete_path, '%s.pcd' % s)
                # target_label_path = os.path.join(target_complete_path, '%s_label.txt' % s)
                o3d.io.write_point_cloud(target_path, pcd)
                # np.savetxt(os.path.join('../datasets/ShapeNet/', 'test.xyz'), y, fmt='%.5f')
                np.savetxt(os.path.join(target_complete_path, '%s-label.txt' % s), cur_label, fmt='%.5f')
        """partial"""
        if gen_partial:
            with open(data_json) as f:
                self.dataset_categories = json.loads(f.read())
            # whole_pcds = collections.defaultdict(list)

            for dc in tqdm(self.dataset_categories, leave=False):
                category_name, s = dc.split('/')[-2], dc.split('/')[-1]
                info = np.loadtxt(os.path.join(shapenet_path, dc+'.txt'), delimiter=' ')[:, :7]
                cur_pcd, cur_normal, cur_label = info[:, :3], info[:, 3:6], info[:, 6]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cur_pcd)

                # for s in tqdm(samples, leave=False):
                #     file = pcd_path % (category_name, s)
                #     info = np.loadtxt(file, delimiter=' ')[:, :7]
                #     cur_pcd, cur_normal, cur_label = info[:, :3], info[:, 3:6], info[:, 6]
                #     whole_pcds[category_name].append(cur_pcd)
                if mode == 'train':
                    target_partial_path = '../datasets/ShapeNetSeg/%s/partial/%s/%s/' % (mode, category_name, s)
                    num_views = 8
                else:
                    target_partial_path = '../datasets/ShapeNetSeg/%s/partial/%s/' % (mode, category_name)
                    num_views = 1
                if not os.path.exists(target_partial_path):
                    os.makedirs(target_partial_path)
                # each view contains 10 unmasked seeds
                num_seed = 64
                for view_idx in range(num_views):
                    # randomly select the seeds
                    rand_idx = np.random.choice(len(cur_pcd), num_seed) # number of seeds (views)
                    seeds = cur_pcd[rand_idx]  # need 8 seeds for training set, 1 for testing set
                    tree = KDTree(cur_pcd, leaf_size=2)
                    dist, ball_idx = tree.query(seeds, k=32)  # each view contains num_seed * 400 points queried
                    masked_pcd, masked_partial_label = [], []
                    for seed_idx in range(len(ball_idx)):
                        masked_patch = cur_pcd[ball_idx[seed_idx]]
                        # masked_patch_label = cur_label[ball_idx[seed_idx]]
                        masked_pcd.append(masked_patch)
                        # masked_partial_label.append(masked_patch_label)
                    masked_pcd = np.concatenate(masked_pcd, 0)
                    # masked_partial_label = np.concatenate(masked_partial_label, 0)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(masked_pcd)
                    target_name = s+'-%d.pcd'%view_idx if mode=='train' else s+'.pcd'
                    o3d.io.write_point_cloud(os.path.join(target_partial_path, target_name), pcd)
                    # target_label_path = os.path.join(target_partial_path, '%s-%d-label.txt' % (s, view_idx))
                    # np.savetxt(target_label_path, masked_partial_label, fmt='%.5f')


def load_h5_data_label_seg(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:] # (2048, 2048, 3)
  label = f['label'][:] # (2048, 1)
  seg = f['pid'][:] # (2048, 2048)
  return (data, label, seg)


class ShapeNetPartV0Processer(object): # extract from original expert annotated shapenet
    def __init__(self, cfg, gen_complete=False, gen_partial=False, mode='train', num_seed=64, k=16, gen_json=False):
        seg_json = '../datasets/ShapeNetPartV0.json'
        category_file = '../datasets/shapenet_part_seg_hdf5_data/synsetoffset2category.txt'
        label_mapping = np.loadtxt(category_file, dtype=str)
        taxonomy_id = 0
        self.cfg = cfg
        seg_dicts = []
        cat_dict = {'train': collections.defaultdict(list), 'test': collections.defaultdict(list)} # 0:train, 1:test
        for m in ('train', 'test'):
            print('Generating mode =', m)
            FILE_LIST = os.path.join('../datasets/shapenet_part_seg_hdf5_data/', '%s_hdf5_file_list.txt' % m)
            file_list = [line.rstrip() for line in open(FILE_LIST)]
            num_file = len(file_list)
            for i in range(num_file):
                cur_filename = os.path.join('../datasets/shapenet_part_seg_hdf5_data/', file_list[i])
                batch_data, batch_label, batch_seg = load_h5_data_label_seg(cur_filename)
                for batch_idx in tqdm(range(len(batch_label))):
                    cur_pcd, cur_label, cur_seg = batch_data[batch_idx, :], batch_label[batch_idx, :], batch_seg[batch_idx, :]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cur_pcd)
                    class_name = label_mapping[cur_label[0]][1]

                    if gen_complete:
                        target_complete_path = '../datasets/ShapeNetPartV0/%s/complete/%s/' % (m, class_name)
                        if not os.path.exists(target_complete_path):
                            os.makedirs(target_complete_path)
                        target_path = os.path.join(target_complete_path, '%d.pcd' % taxonomy_id)
                        o3d.io.write_point_cloud(target_path, pcd)
                        np.savetxt(os.path.join(target_complete_path, '%d.txt' % taxonomy_id), cur_seg, fmt='%.5f')
                    """partial"""
                    if gen_partial:

                        if m == 'train':
                            target_partial_path = '../datasets/ShapeNetPartV0/%s/partial/%s/%d/' % (m, class_name, taxonomy_id)
                            num_views = 8
                        else:
                            target_partial_path = '../datasets/ShapeNetPartV0/%s/partial/%s/' % (m, class_name)
                            num_views = 1
                        if not os.path.exists(target_partial_path):
                            os.makedirs(target_partial_path)
                        # each view contains 10 unmasked seeds
                        # num_seed = num_seed
                        for view_idx in range(num_views):
                            # randomly select the seeds
                            rand_idx = np.random.choice(len(cur_pcd), num_seed) # number of seeds (views)
                            seeds = cur_pcd[rand_idx]  # need 8 seeds for training set, 1 for testing set
                            tree = KDTree(cur_pcd, leaf_size=2)
                            dist, ball_idx = tree.query(seeds, k=k)  # each view contains num_seed * 400 points queried
                            masked_pcd, masked_partial_label = [], []
                            for seed_idx in range(len(ball_idx)):
                                masked_patch = cur_pcd[ball_idx[seed_idx]]
                                # masked_patch_label = cur_label[ball_idx[seed_idx]]
                                masked_pcd.append(masked_patch)
                                # masked_partial_label.append(masked_patch_label)
                            masked_pcd = np.concatenate(masked_pcd, 0)
                            # masked_partial_label = np.concatenate(masked_partial_label, 0)
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(masked_pcd)
                            target_name = '%d-%d.pcd' % (taxonomy_id, view_idx) if m=='train' else '%d.pcd' % taxonomy_id
                            o3d.io.write_point_cloud(os.path.join(target_partial_path, target_name), pcd)
                    """counting for json generation"""
                    cat_dict[m][class_name].append(str(taxonomy_id))
                    taxonomy_id += 1
        if gen_json:
            for category, category_id in label_mapping:
                new_dict = {
                    'taxonomy_id': category_id,
                    'taxonomy_name': category,
                    'test': cat_dict['test'][category_id],
                    'train': cat_dict['train'][category_id],
                }
                seg_dicts.append(new_dict)
            with open(seg_json, "w") as outfile:
                json.dump(seg_dicts, outfile, indent=4)


def load_h5(path, BG):
    # BG = False if '_nobg' in path else True
    f = h5py.File(path)
    data = f['data'][:]
    label = f['label'][:]
    if BG:
        mask = f['mask'][:]
    print(data.shape, label.shape)
    if BG:
        return data, label, mask
    return data, label
        

class ScanOjbectProcesser(object):
    def __init__(self, cfg, gen_complete=False, gen_partial=False, num_seed=64, k=16, BG=True, modes=('test', 'train')):
        obj_id = 0
        self.cfg = cfg
        BG_postfix = '' if BG else '_nobg'
        tar_type = '%s_75_s%d_k%d' % (BG_postfix, num_seed, k) #'_bg_75'
        category_file = '../datasets/ScanObjectNN%s/scanobjectnn_shape_names.txt' % tar_type
            
        for m in modes:
            obj_file = '../datasets/ScanObjectNN%s/scanobjectnn_%s.txt' % (tar_type, m)
            mode_objs = []
            print('Generating mode =', m)
            m_ = 'training' if m == 'train' else m
            # BG_postfix = '' if BG else '_nobg'
            h5_path = '../datasets/h5_files/main_split%s/%s_objectdataset_augmentedrot_scale75.h5' % (BG_postfix, m_)
            # BG = False if  in h5_path else True
            print('Background?', BG)
            if BG:
                data, label, _ = load_h5(h5_path, BG)
            else:
                data, label = load_h5(h5_path, BG)
            num_file = len(label) # how many objects
            for i in tqdm(range(num_file)):
                cur_pcd, cur_label = data[i], label[i]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cur_pcd)
                class_name = cur_label

                if gen_complete:
                    target_complete_path = '../datasets/ScanObjectNN%s/%s/complete/%02d/' % (tar_type, m, class_name)
                    if not os.path.exists(target_complete_path):
                        os.makedirs(target_complete_path)
                    target_path = os.path.join(target_complete_path, '%02d_%05d.pcd' % (class_name, obj_id))
                    o3d.io.write_point_cloud(target_path, pcd)
                    
                """partial"""
                if gen_partial:
                    if m == 'train':
                        target_partial_path = '../datasets/ScanObjectNN%s/%s/partial/%02d/%02d_%05d/' % (tar_type, m, class_name, class_name, obj_id)
                        num_views = 8
                    else:
                        target_partial_path = '../datasets/ScanObjectNN%s/%s/partial/%02d/' % (tar_type, m, class_name)
                        num_views = 1
                    if not os.path.exists(target_partial_path):
                        os.makedirs(target_partial_path)
                    # each view contains x unmasked seeds
                    # num_seed = 64
                    for view_idx in range(num_views):
                        # randomly select the seeds
                        rand_idx = np.random.choice(len(cur_pcd), num_seed) # number of seeds (views)
                        seeds = cur_pcd[rand_idx]  # need 8 seeds for training set, 1 for testing set
                        tree = KDTree(cur_pcd, leaf_size=2)
                        dist, ball_idx = tree.query(seeds, k=k)  # each view contains num_seed * 400 points queried
                        masked_pcd, masked_partial_label = [], []
                        for seed_idx in range(len(ball_idx)):
                            masked_patch = cur_pcd[ball_idx[seed_idx]]
                            # masked_patch_label = cur_label[ball_idx[seed_idx]]
                            masked_pcd.append(masked_patch)
                            # masked_partial_label.append(masked_patch_label)
                        masked_pcd = np.concatenate(masked_pcd, 0)
                        # masked_partial_label = np.concatenate(masked_partial_label, 0)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(masked_pcd)
                        target_name = '%02d_%05d-%d.pcd' % (class_name, obj_id, view_idx) if m=='train' else '%02d_%05d.pcd' % (class_name, obj_id)
                        o3d.io.write_point_cloud(os.path.join(target_partial_path, target_name), pcd)
                """counting for json generation"""
                mode_objs.append(str('%02d_%05d' % (class_name, obj_id)))                
                obj_id += 1
            np.savetxt(obj_file, mode_objs, fmt='%s')
        np.savetxt(category_file, list(set(label)), fmt='%02d')


if __name__ == '__main__':
    from sklearn.neighbors import KDTree
    # loader = ModelNetDataLoader(cfg, gen_complete=False, gen_partial=True, num_seed=64, k=4)
    # loader = ScanOjbectProcesser(cfg, gen_complete=True, gen_partial=False, num_seed=4, k=128, BG=True)
    # loader = ShapeNetProcesser(cfg, gen_complete=True, gen_partial=False, mode='test') # mode= test or train
    # loader = ShapeNetProcesser2(cfg, gen_complete=True, gen_partial=False, mode='train') # mode= test or train
    loader = ShapeNetPartV0Processer(cfg, gen_complete=False, gen_partial=True, mode='train', num_seed=64, k=16, gen_json=True) # mode= test or train
    # category_file = '../datasets/shapenet_part_seg_hdf5_data/synsetoffset2category.txt'
    # x = np.loadtxt(category_file, dtype=str)
    # print(x[0])

    # pcd_name = '/home/kz/Documents/SnowflakeNet/datasets/PCN/train/partial/02933112/1a2c199d7c8cb7f83d724fd1eb6db6b9/01.pcd'
    # pcd_name = '/home/kz/Documents/SnowflakeNet/datasets/PCN/train/complete/02933112/1a2c199d7c8cb7f83d724fd1eb6db6b9.pcd'
    # pcd_name = '../datasets/ModelNet40/test/partial64*8/airplane/airplane_0627.pcd'
    # pcd = read_ply(pcd_name)
    # np.savetxt('../test12.xyz', pcd)
