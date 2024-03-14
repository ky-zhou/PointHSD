import logging
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.loss_utils import *
from models.model25 import SnowflakeNet as Model
import numpy as np
from core.visualize import *
import sklearn.metrics as metrics


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
        # model = Model(dim_feat=512, num_p0=1024, up_factors=[1, 1])
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
    total_kl_r1, total_kl_r2, total_kl_r3 = 0, 0, 0
    total_mse_1, total_mse_2, total_mse_3 = 0, 0, 0
    n_batches = len(test_data_loader)

    test_accs, mAccs, entropies = [], [], [] # for all categories
    test_trues = {0: [], 1: [], 2:[]}
    test_preds = {0: [], 1: [], 2:[]}
    test_feats = {0: [], 1: [], 2:[]}
    feat_npy1, feat_npy2, feat_npy3, label_npy = [], [], [], []
    logit_npy1, logit_npy2, logit_npy3 = [], [], []
    X_prime = []
    x_train, x_mid, wb, y_onehot, y = [], [], [], [], []
    counter, indices, last_idx = collections.Counter(), [], 2
    # Testing loop
    with tqdm(test_data_loader) as t: # each batch seems to have only one taxonomy, at least in test set
        for model_idx, (taxonomy_id, model_id, data, gt_label) in enumerate(t):
            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']
                gt_label = utils.helpers.var_or_cuda(torch.Tensor(gt_label).long())
                # print(f'test partial {partial.shape}')
                x_train_bat = partial.detach().cpu().numpy()#.reshape(len(taxonomy_ids), -1)
                x_train.append(x_train_bat)
                y_onehot_bat = torch.nn.functional.one_hot(gt_label, 15).detach().cpu().numpy()
                y_onehot.append(y_onehot_bat)
                y.append(gt_label.detach().cpu().numpy())

                b, n, _ = partial.shape

                pcds_pred, labels_pred, feats_cls = model(partial.contiguous())
                for i, logit in enumerate(labels_pred):
                    pred = logit.argmax(-1)
                    test_trues[i].append(gt_label.cpu().detach().numpy())
                    test_preds[i].append(pred.cpu().detach().numpy())
                    # test_feats[i].append(feats_cls[i+1].cpu().detach().numpy())

                if calc_loss:
                    loss_total, losses, cur_idx, best_idx = get_loss_nomi(labels_pred, gt_label, pcds_pred, partial, gt, feats_cls,
                                                              last_idx, indices, epoch_idx, mse=cfg.TRAIN.CODE, nomi=cfg.TRAIN.NOMI)

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
                    ce_s1_item = losses[5].item() * 1e2
                    total_ce_s1 += ce_s1_item
                    ce_s2_item = losses[6].item() * 1e2
                    total_ce_s2 += ce_s2_item
                    ce_s3_item = losses[7].item() * 1e2
                    total_ce_s3 += ce_s3_item
                    kl_r1_item = losses[8].item()# * 1e2
                    total_kl_r1 += kl_r1_item
                    kl_r2_item = losses[9].item()# * 1e2
                    total_kl_r2 += kl_r2_item

                if show and epoch_idx % cfg.TRAIN.SAVE_FREQ == 0:
                    visual_data = {}
                    visual_data['pred'] = pcds_pred[-1].cpu()
                    visual_data['part'] = partial.cpu()
                    visual_data['original'] = gt.cpu()
                    plot_fig(visual_data, save_path, model_idx)

                """save feat"""
                feat_npy1.append(feats_cls[1].cpu().detach().numpy())
                feat_npy2.append(feats_cls[2].cpu().detach().numpy())
                feat_npy3.append(feats_cls[3].cpu().detach().numpy())
                logit_npy1.append(labels_pred[0].cpu().detach().numpy())
                logit_npy2.append(labels_pred[1].cpu().detach().numpy())
                logit_npy3.append(labels_pred[2].cpu().detach().numpy())
                label_npy.append(gt_label.cpu().detach().numpy())
                X_prime.append(pcds_pred[-1].cpu().detach().numpy())
                """save data"""
                method_name = 'pointnet-hsd-snn-s128-k8'
                # for idx in range(len(model_id)):
                #     cat_dir = os.path.join(cfg.DIR.OUT_PATH, 'data-%s/%s' % (method_name, model_id[idx].item()))  # only used for testing
                #     if not os.path.exists(cat_dir):
                #         os.makedirs(cat_dir)
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

    for j in range(len(labels_pred)):
        test_true = np.concatenate(test_trues[j])
        test_pred = np.concatenate(test_preds[j])
        test_accs.append(metrics.accuracy_score(test_true, test_pred))
        mAccs.append(metrics.balanced_accuracy_score(test_true, test_pred))

    #For information plane
    method_name = 'pointnet-hsd-snn-s128-k8'
    feat_dir = os.path.join(cfg.DIR.OUT_PATH, 'data-%s' % method_name)  # only used for testing
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    x_train = np.concatenate(x_train)
    print(f'x_train: {x_train.shape}')
    y = np.concatenate(y)
    y_onehot = np.concatenate(y_onehot)
    print(f'y_onehot and y: {y_onehot.shape, y.shape}')
    x_prime = np.concatenate(model.module.decoder.pcd_out)
    print(f'X_prime: {x_prime.shape}')
    model.module.decoder.deep_cls.gather_info()
    weights = model.module.decoder.deep_cls.repres
    logits = model.module.decoder.deep_cls.logits
    xyzs = model.module.decoder.deep_cls.xyzs
    print(f'xyzs, weights and logits: {xyzs[1].shape, weights[1].shape, logits[1].shape}')
    np.save(os.path.join(feat_dir, 'xyz-1-%d.npy' % epoch_idx), xyzs[1])
    np.save(os.path.join(feat_dir, 'xyz-2-%d.npy' % epoch_idx), xyzs[2])
    np.save(os.path.join(feat_dir, 'xyz-3-%d.npy' % epoch_idx), xyzs[3])
    # np.save(os.path.join(feat_dir, 'feat-1-%d.npy' % epoch_idx), weights[1])
    # np.save(os.path.join(feat_dir, 'feat-2-%d.npy' % epoch_idx), weights[2])
    # np.save(os.path.join(feat_dir, 'feat-3-%d.npy' % epoch_idx), weights[3])
    np.save(os.path.join(feat_dir, 'logit-1-%d.npy' % epoch_idx), logits[1])
    np.save(os.path.join(feat_dir, 'logit-2-%d.npy' % epoch_idx), logits[2])
    np.save(os.path.join(feat_dir, 'logit-3-%d.npy' % epoch_idx), logits[3])
    np.save(os.path.join(feat_dir, 'x_prime-%d.npy' % epoch_idx), x_prime)
    if epoch_idx == 1:
        np.save(os.path.join(feat_dir, 'x_train-%d.npy' % epoch_idx), x_train)
        np.save(os.path.join(feat_dir, 'label-%d.npy' % epoch_idx), y)
        np.save(os.path.join(feat_dir, 'labelonehot-%d.npy' % epoch_idx), y_onehot)
    model.module.decoder.deep_cls.reset()
    model.module.decoder.reset()

    """Information plane"""
    feat_dir = os.path.join(cfg.DIR.OUT_PATH, 'data-%s' % method_name)  # only used for testing
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    feat_npy1 = np.concatenate(feat_npy1)
    feat_npy2 = np.concatenate(feat_npy2)
    feat_npy3 = np.concatenate(feat_npy3)
    logit_npy1 = np.concatenate(logit_npy1)
    logit_npy2 = np.concatenate(logit_npy2)
    logit_npy3 = np.concatenate(logit_npy3)
    label_npy = np.concatenate(label_npy)
    X_prime = np.concatenate(X_prime)
    np.save(os.path.join(feat_dir, 'feat1-%d.npy' % epoch_idx), feat_npy1)
    np.save(os.path.join(feat_dir, 'feat2-%d.npy' % epoch_idx), feat_npy2)
    np.save(os.path.join(feat_dir, 'feat3-%d.npy' % epoch_idx), feat_npy3)
    np.save(os.path.join(feat_dir, 'logit1-%d.npy' % epoch_idx), logit_npy1)
    np.save(os.path.join(feat_dir, 'logit2-%d.npy' % epoch_idx), logit_npy2)
    np.save(os.path.join(feat_dir, 'logit3-%d.npy' % epoch_idx), logit_npy3)
    if epoch_idx == 1:
        np.save(os.path.join(feat_dir, 'label.npy'), label_npy)
    np.save(os.path.join(feat_dir, 'X_prime-%d.npy' % epoch_idx), X_prime)

    avg_cd3 = 0
    if calc_loss:
        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches
        avg_cd3 = total_cd_p3 / n_batches
        avg_partial = total_partial / n_batches
        avg_ce1 = total_ce_s1 / n_batches
        avg_ce2 = total_ce_s2 / n_batches
        avg_ce3 = total_ce_s3 / n_batches
        avg_kl1 = total_kl_r1 / n_batches
        avg_kl2 = total_kl_r2 / n_batches
        if cfg.TRAIN.CODE:
            avg_mse1 = total_mse_1 / n_batches
            test_writer.add_scalar('Loss/Epoch/total_mse_1', avg_mse1, epoch_idx)
            avg_mse2 = total_mse_2 / n_batches
            test_writer.add_scalar('Loss/Epoch/total_mse_2', avg_mse2, epoch_idx)
    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/TEST/Acc1', test_accs[0], epoch_idx)
        test_writer.add_scalar('Loss/TEST/Acc2', test_accs[1], epoch_idx)
        test_writer.add_scalar('Loss/TEST/Acc3', test_accs[2], epoch_idx)
        # test_writer.add_scalar('Loss/Epoch/Acc4', test_accs.avg(3), epoch_idx)
        test_writer.add_scalar('Loss/TEST/mAcc1', mAccs[0], epoch_idx)
        test_writer.add_scalar('Loss/TEST/mAcc2', mAccs[1], epoch_idx)
        test_writer.add_scalar('Loss/TEST/mAcc3', mAccs[2], epoch_idx)
        if calc_loss:
            test_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/cd_p3', avg_cd3, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/partial_matching', avg_partial, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/ce_s1', avg_ce1, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/ce_s2', avg_ce2, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/ce_s3', avg_ce3, epoch_idx)
            # test_writer.add_scalar('Loss/Epoch/best_idx', best_idx, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/kl_r1', avg_kl1, epoch_idx)
            test_writer.add_scalar('Loss/Epoch/kl_r2', avg_kl2, epoch_idx)

    print(f'Overall: {test_accs}, mean: {mAccs}, CD: {avg_cd3}, entropy: {entropies}')

    return test_accs, mAccs, avg_cd3
