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
from core.test_18 import test_net
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import *
# from models.curvenet_cls import CurveNet as Model
# from models.model18 import PointMLP as Model
from models.model18 import PointNet as Model
# from models.model18 import CurveNet as Model
# from torch.optim.lr_scheduler import CosineAnnealingLR


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    train_data_loader = torch.utils.data.DataLoader(utils.data_loaders.ModelNet40H5(partition='train', num_points=1024), num_workers=cfg.CONST.NUM_WORKERS,
                              batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(utils.data_loaders.ModelNet40H5(partition='test', num_points=1024), num_workers=cfg.CONST.NUM_WORKERS//2,
                             batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last=False)

    # Set up folders for logs and checkpoints
    show_fig = False
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)
    cfg.DIR.FIGPATH = output_dir % 'figs'
    if not os.path.exists(cfg.DIR.FIGPATH) and show_fig:
        os.makedirs(cfg.DIR.FIGPATH)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # model = Model(40)
    model = Model(51) # uni
    # model = Model(points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
    #                activation="relu", bias=False, use_xyz=False, normalize="anchor",
    #                dim_expansion=[2, 2, 2], pre_blocks=[2, 2, 2], pos_blocks=[2, 2, 2],
    #                k_neighbors=[8, 16, 24], reducers=[2, 2, 2])
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)

    # lr scheduler
    # scheduler_steplr = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    # scheduler_steplr = ReduceLROnPlateau(optimizer, factor=cfg.TRAIN.GAMMA, min_lr=0.00001)
    # lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
    #                                       after_scheduler=scheduler_steplr)

    lr_scheduler = CosineAnnealingLR(optimizer, 300, eta_min=0.005)
    init_epoch = 0
    best_metrics, best_metrics_mean = float('-inf'), float('-inf')
    steps = 0

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        best_metrics = checkpoint['best_metrics']
        model.load_state_dict(checkpoint['model'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        total_ce_d, total_ce_s1, total_ce_s2, total_ce_s3 = 0, 0, 0, 0
        total_kl_r1, total_kl_r2, total_kl_r3 = 0, 0, 0
        total_mse = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        with tqdm(train_data_loader) as t:
            for batch_idx, (data, gt_label) in enumerate(t):
                # print('taxonomy_ids:', taxonomy_ids)
                data_time.update(time() - batch_end_time)
                data = utils.helpers.var_or_cuda(data)
                gt_label = utils.helpers.var_or_cuda(gt_label.squeeze())
                # print('train:', data.shape, gt_label.shape)

                labels_pred, _ = model(data.permute(0, 2, 1).contiguous())

                loss_total, losses = get_loss_1ce(labels_pred, gt_label)

                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.8)

                accs = []
                pred = labels_pred.argmax(-1)
                acc = (pred == gt_label).sum() / float(gt_label.size(0))
                accs.append(acc)

                ce_s1_item = losses[0].item() * 1e2
                total_ce_s1 += ce_s1_item
                # mse_item = losses[5].item()# * 1e2
                # total_mse += mse_item
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description('[Epoch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS))
                t.set_postfix(acc='%s' % ['%.4f' % acc for acc in accs])
                # t.set_postfix(acc='%s' % ['%.4f' % acc for acc in accs],
                #               kl='%s' % ['%.4f' % kl for kl in [kl_r1_item, kl_r2_item]])

                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

        avg_ce1 = total_ce_s1 / n_batches
        # avg_mse = total_mse / n_batches

        lr_scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/ce_s1', avg_ce1, epoch_idx)
        # train_writer.add_scalar('Loss/Epoch/total_mse', avg_mse, epoch_idx)
        train_writer.add_scalar('Learning_Rate/Hyper/lr', optimizer.param_groups[0]['lr'], epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses_CE = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time,
             ['%.4f' % l for l in [avg_ce1]]))
             # ['%.4f' % l for l in [avg_ce1, avg_ce2, avg_ce3]]))

        # Validate the current model
        best_acc, best_mean = test_net(cfg, epoch_idx, val_data_loader, val_writer, model, show=show_fig, save_path=cfg.DIR.FIGPATH)

        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0:
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, 'ckpt-epoch-%03d.pth' % epoch_idx)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'best_metrics_mean': best_mean,
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
                'scheduler': lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict()
            }, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

        if best_acc > best_metrics:
            best_metrics = best_acc
        if best_mean > best_metrics_mean:
            best_metrics_mean = best_mean
        print(f'Best acc: {best_metrics}, mean: {best_metrics_mean}')

    train_writer.close()
    val_writer.close()
