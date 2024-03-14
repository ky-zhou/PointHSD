# -*- coding: utf-8 -*-
# @Author: XP

from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.COMPLETION3D                        = edict()
__C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/path/to/datasets/Completion3D/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/path/to/datasets/Completion3D/%s/gt/%s/%s.h5'
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 8
__C.DATASETS.SHAPENET.N_POINTS                   = 1024
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = './datasets/PCN/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = './datasets/PCN/%s/complete/%s/%s.pcd'
__C.DATASETS.SHAPENET.SEG_PARTIAL_POINTS_PATH     = './datasets/ShapeNetPartV0/%s/partial/%s/%s/%s-%d.pcd'
__C.DATASETS.SHAPENET.SEG_COMPLETE_POINTS_PATH    = './datasets/ShapeNetPartV0/%s/complete/%s/%s.pcd'
__C.DATASETS.SHAPENET.SEGMENTATION_FILE_PATH  = './datasets/ShapeNetPartV0.json'
__C.DATASETS.SHAPENET.PARTIAL_LABELS_PATH     = './datasets/ShapeNetPartV0/%s/partial/%s/%s-%d.txt'
__C.DATASETS.SHAPENET.COMPLETE_LABELS_PATH    = './datasets/ShapeNetPartV0/%s/complete/%s/%s.txt'
__C.DATASETS.MODELNET                            = edict()
__C.DATASETS.MODELNET.CATEGORY_FILE_PATH         = './datasets/ModelNet40/modelnet40_shape_names.txt'
__C.DATASETS.MODELNET.N_RENDERINGS               = 8
__C.DATASETS.MODELNET.N_POINTS                   = 512
__C.DATASETS.MODELNET.PARTIAL_POINTS_PATH        = './datasets/ModelNet40/%s/partial/%s/%s/%s-%d.pcd'
__C.DATASETS.MODELNET.COMPLETE_POINTS_PATH       = './datasets/ModelNet40/%s/complete/%s/%s.pcd'
__C.DATASETS.SCANOBNN                            = edict()
__C.DATASETS.SCANOBNN.TYPE                       = 'ScanObjectNN_nobg_75_s64_k16'
__C.DATASETS.SCANOBNN.CATEGORY_FILE_PATH         = './datasets/'+__C.DATASETS.SCANOBNN.TYPE+'/scanobjectnn_shape_names.txt'
__C.DATASETS.SCANOBNN.N_RENDERINGS               = 8
__C.DATASETS.SCANOBNN.N_POINTS                   = 1024
__C.DATASETS.SCANOBNN.PARTIAL_POINTS_PATH        = './datasets/'+__C.DATASETS.SCANOBNN.TYPE+'/%s/partial/%s/%s/%s-%d.pcd'
__C.DATASETS.SCANOBNN.COMPLETE_POINTS_PATH       = './datasets/'+__C.DATASETS.SCANOBNN.TYPE+'/%s/complete/%s/%s.pcd'
#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: ModelNet40, ScanObjectNN, ModelNet10, ShapeNetPart
__C.DATASET.TRAIN_DATASET                        = 'ScanObjectNN'
__C.DATASET.TEST_DATASET                         = 'ScanObjectNN'

#
# Constants
#
__C.CONST                                        = edict()


#
# Directories
#

__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './output'
__C.CONST.NUM_WORKERS                            = 20
__C.CONST.DEVICE                                 = '0,1,2,3,4,5,6,7,8,9'#
__C.CONST.N_INPUT_POINTS                         = 2048
# __C.CONST.WEIGHTS                                = './output/checkpoints/2023-08-14T17:54:32.027133/ckpt-best.pth' #
# __C.CONST.WEIGHTS                                = './output/checkpoints/2023-08-13T08:37:32.494599/ckpt-best-ciou.pth' #
# __C.CONST.WEIGHTS                                = './output/checkpoints/2023-08-03T09:44:26.457782/ckpt-epoch-100.pth' #

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 260 # never 240 for mn40 (bug)
__C.TRAIN.N_EPOCHS                               = 300
__C.TRAIN.SAVE_FREQ                              = 100
__C.TRAIN.LEARNING_RATE                          = 0.001
__C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
__C.TRAIN.LR_DECAY_STEP                          = 50
__C.TRAIN.WARMUP_STEPS                           = 50
__C.TRAIN.GAMMA                                  = .8 # default .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.NOMI                                   = False
__C.TRAIN.CODE                                   = False
#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
