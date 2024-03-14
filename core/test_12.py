import logging
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.loss_utils import *
from models.model18 import SnowflakeNet as Model
import numpy as np
from core.visualize import *
import sklearn.metrics as metrics
import open3d as o3d


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None, show=False, save_path=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    calc_loss = False

    if test_data_loader is None:
        # Set up data loader
        if cfg.DATASET.TRAIN_DATASET == 'ShapeNet':
            collate_fn = utils.data_loaders.collate_fn
            ncat = 8
        elif cfg.DATASET.TRAIN_DATASET == 'ModelNet40':
            collate_fn = utils.data_loaders.collate_fn2
            ncat = 40
        elif cfg.DATASET.TRAIN_DATASET == 'ScanObjectNN':
            collate_fn = utils.data_loaders.collate_fn3
            ncat = 15
        else:
            raise(NotImplementedError)
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
                                                       batch_size=cfg.TRAIN.BATCH_SIZE,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
                                                       collate_fn=collate_fn,
                                                       pin_memory=True,
                                                       shuffle=False,
                                                       drop_last=False)

    # Setup networks and initialize networks
    if model is None:
        model = Model(dim_feat=512, ncat=ncat, up_factors=[1, 2])
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        save_path = cfg.CONST.WEIGHTS.replace('checkpoints', 'figs').rstrip('/ckpt-best.pth')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        calc_loss = True

    calc_loss = True
    # Switch models to evaluation mode
    model.eval()

    total_cd_pc = 0
    total_cd_p1 = 0
    total_cd_p2 = 0
    total_cd_p3 = 0
    total_partial = 0
    total_ce_d, total_ce_s1, total_ce_s2, total_ce_s3 = 0, 0, 0, 0
    n_batches = len(test_data_loader)

    test_true = []
    test_pred = []
    feat_npy, label_npy = [], []
    with torch.no_grad():
    # Testing loop
        with tqdm(test_data_loader) as t: # each batch seems to have only one taxonomy, at least in test set
            for model_idx, (taxonomy_id, model_id, data, gt_label) in enumerate(t):
                # print(f'taxonomy_id, model_id {taxonomy_id, model_id}')

                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']
                gt_label = utils.helpers.var_or_cuda(torch.Tensor(gt_label).long())
                # print(f'test partial {partial.shape}')
                # print(f'test gt_label {gt_label.shape}') # 64

                b, n, _ = partial.shape

                pcds_pred, labels_pred, feats_cls = model(partial.contiguous())

                if calc_loss:
                    loss_total, losses = get_loss_up_1ce(pcds_pred, labels_pred, partial, gt, gt_label, feats_cls, sqrt=True)

                    cd_pc_item = losses[0].item() * 1e3
                    total_cd_pc += cd_pc_item
                    cd_p1_item = losses[1].item() * 1e3
                    total_cd_p1 += cd_p1_item
                    cd_p2_item = losses[2].item() * 1e3
                    total_cd_p2 += cd_p2_item
                    cd_p3_item = losses[3].item() * 1e3
                    total_cd_p3 += cd_p3_item
                    partial_item = losses[4].item() * 1e3
                    total_partial += partial_item
                    ce_d_item = losses[5].item() * 1e2
                    total_ce_d += ce_d_item

                pred = labels_pred.argmax(-1)
                test_true.append(gt_label.cpu().detach().numpy())
                test_pred.append(pred.cpu().detach().numpy())
                """save feat"""
                # feat_npy.append(feats_cls.cpu().detach().numpy())
                # label_npy.append(gt_label.cpu().detach().numpy())

                if show and epoch_idx % cfg.TRAIN.SAVE_FREQ == 0:
                    visual_data = {}
                    visual_data['pred'] = pcds_pred[-1].cpu()
                    visual_data['part'] = partial.cpu()
                    visual_data['original'] = gt.cpu()
                    plot_fig(visual_data, save_path, model_idx)

                # method_name = 'pointnet'
                # for idx in range(len(model_id)):
                #     cat_dir = os.path.join(cfg.DIR.OUT_PATH, 'data-%s/%s' % (method_name, model_id[idx].item()))  # only used for testing
                #     if not os.path.exists(cat_dir):
                #         os.makedirs(cat_dir)
                #     """Save to pcd"""
                #     """Save to xyz or npy"""
                #     # file_path = os.path.join(cat_dir, '%s_%d_in.xyz' % (taxonomy_id, model_id))
                #     # np.savetxt(file_path, partial.data.cpu().numpy()[idx, ...], fmt='%.6f', delimiter=' ')
                #     # file_path = os.path.join(cat_dir, '%s_%d_%d.xyz' % (model_id[idx].item(), model_idx, idx))
                #     # np.savetxt(file_path, pcds_pred[-1].data.cpu().numpy()[idx, ...], fmt='%.6f', delimiter=' ')
                #     # file_path = os.path.join(cat_dir, '%s_%d_%d_gt.xyz' % (model_id[idx].item(), model_idx, idx))
                #     # np.savetxt(file_path, gt.data.cpu().numpy()[idx, ...], fmt='%.6f', delimiter=' ')
                #     """Save to npy"""
                #     file_path = os.path.join(cat_dir, '%s_%d_%d_in.npy' % (model_id[idx].item(), model_idx, idx))
                #     np.save(file_path, partial.data.cpu().numpy()[idx, ...])
                #     file_path = os.path.join(cat_dir, '%s_%d_%d_pred.npy' % (model_id[idx].item(), model_idx, idx))
                #     np.save(file_path, pcds_pred[-1].data.cpu().numpy()[idx, ...])
                #     file_path = os.path.join(cat_dir, '%s_%d_%d_gt.npy' % (model_id[idx].item(), model_idx, idx))
                #     np.save(file_path, gt.data.cpu().numpy()[idx, ...])



    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_accs = metrics.accuracy_score(test_true, test_pred)
    mAcc = metrics.balanced_accuracy_score(test_true, test_pred)
    # feat_dir = os.path.join(cfg.DIR.OUT_PATH, 'data-%s' % method_name)  # only used for testing
    # if not os.path.exists(feat_dir):
    #     os.makedirs(feat_dir)
    # feat_npy = np.concatenate(feat_npy)
    # label_npy = np.concatenate(label_npy)
    # np.save(os.path.join(feat_dir, 'feat-pn.npy'), feat_npy)
    # np.save(os.path.join(feat_dir, 'label-pn.npy'), label_npy)

    if calc_loss:
        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches
        avg_cd3 = total_cd_p3 / n_batches
        avg_partial = total_partial / n_batches
        avg_ced = total_ce_d / n_batches
    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/TEST/Acc1', test_accs, epoch_idx)
        test_writer.add_scalar('Loss/TEST/mAcc1', mAcc, epoch_idx)
        if calc_loss:
            test_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/cd_p3', avg_cd3, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/partial_matching', avg_partial, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/ce_d', avg_ced, epoch_idx)
    print(f'Overall: {test_accs}, mean: {mAcc}, CD: {avg_cd3}')

    return test_accs, mAcc, avg_cd3

