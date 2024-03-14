# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import torch
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
from torch.utils.tensorboard import SummaryWriter
from core.test_12 import test_net
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import *
from models.model18 import SnowflakeNet as Model
from core.visualize import plot_fig


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
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
    
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False,
                                                  drop_last=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)
    cfg.DIR.FIGPATH = output_dir % 'figs'
    if not os.path.exists(cfg.DIR.FIGPATH):
        os.makedirs(cfg.DIR.FIGPATH)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    model = Model(dim_feat=512, num_pc=256, ncat=ncat, up_factors=[1, 2])
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    total_params1 = sum(p.numel() for p in model.parameters())
    total_params2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {total_params1, total_params2}')
    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)

    # lr scheduler
    scheduler_steplr = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    # scheduler_steplr = ReduceLROnPlateau(optimizer, factor=cfg.TRAIN.GAMMA, min_lr=0.00001)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)

    init_epoch = 0
    best_metrics, best_metrics_mean = float('-inf'), float('-inf')
    best_metrics_CD = float('inf')
    steps = 0

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_index']
        best_metrics = checkpoint['best_metrics']
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0
        total_cd_p3 = 0
        total_partial = 0
        total_ce_d, total_ce_s1, total_ce_s2, total_ce_s3 = 0, 0, 0, 0
        total_kl_r1, total_kl_r2, total_kl_r3 = 0, 0, 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data, gt_label) in enumerate(t):
                # print('taxonomy_ids:', taxonomy_ids)
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']#[:, :4096, :].contiguous()
                # print('train:', partial.shape, gt.shape)
                gt_label = utils.helpers.var_or_cuda(torch.Tensor(gt_label).long())

                pcds_pred, labels_pred, feats_cls = model(partial)
                # print('train:', pcds_pred[-1].shape)

                loss_total, losses = get_loss_up_1ce(pcds_pred, labels_pred, partial, gt, gt_label, feats_cls, sqrt=True)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                accs = []
                # for logit in labels_pred:
                pred = labels_pred.argmax(-1)
                acc = (pred == gt_label).sum() / float(gt_label.size(0))
                accs.append(acc)

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
                # n_itr = (epoch_idx - 1) * n_batches + batch_idx
                # for acc_idx, acc in enumerate(accs):
                #     train_writer.add_scalar('Metric/Batch/acc%s' % str(acc_idx+1), acc, n_itr)
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description('[Epoch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS))
                t.set_postfix(cd='%s' % ['%.4f' % l for l in [cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item]],
                              acc='%s' % ['%.4f' % acc for acc in accs])

                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches
        avg_cd3 = total_cd_p3 / n_batches
        avg_partial = total_partial / n_batches
        avg_ced = total_ce_d / n_batches

        lr_scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p3', avg_cd3, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/partial_matching', avg_partial, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/ce_d', avg_ced, epoch_idx)
        train_writer.add_scalar('Hyper/Epoch/lr', optimizer.param_groups[0]['lr'], epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s Losses_CE = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]],
             ['%.4f' % l for l in [avg_ced]]))

        # Validate the current model
        best_acc, best_mean, best_CD = test_net(cfg, epoch_idx, val_data_loader, val_writer, model, show=True, save_path=cfg.DIR.FIGPATH)

        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0:
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, 'ckpt-epoch-%03d.pth' % epoch_idx)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'best_metrics_mean': best_mean,
                'best_metrics_CD': best_CD,
                'scheduler': lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict()
            }, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

        if best_acc > best_metrics and epoch_idx > 50:
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, 'ckpt-best.pth')
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'best_metrics_mean': best_mean,
                'best_metrics_CD': best_CD,
                'scheduler': lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict()
            }, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

        if best_acc > best_metrics:
            best_metrics = best_acc
        if best_mean > best_metrics_mean:
            best_metrics_mean = best_mean
        if best_CD < best_metrics_CD:
            best_metrics_CD = best_CD
        print(f'Best acc: {best_metrics}, mean: {best_metrics_mean}, CD: {best_metrics_CD}')

    train_writer.close()
    val_writer.close()
