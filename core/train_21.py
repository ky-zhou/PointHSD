import logging
import os
import torch
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
import collections
from torch.utils.tensorboard import SummaryWriter
from core.test_21 import test_net
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import *
from models.model21 import SnowflakeNet as Model

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
                                                    collate_fn=collate_fn, #2: MN40,3:SCAN
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS,
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
    # log_file = open(os.path.join(cfg.DIR.LOGS, 'logs.txt'), 'w')

    # model = Model(dim_feat=512, num_pc=256, num_p0=1024, up_factors=[1, 1])
    model = Model(dim_feat=512, num_pc=256, dim_cat=ncat, up_factors=[1, 2])
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
    # lr_scheduler = CosineAnnealingLR(optimizer, 300, eta_min=0.005)

    init_epoch = 0
    best_metrics, best_metrics_mean = [float('-inf')]*3, [float('-inf')]*3
    best_metrics_CD = float('inf')
    steps = 0
    counter, indices, last_idx = collections.Counter(), [], 2

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

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0
        total_cd_p3 = 0
        total_partial = 0
        total_ce_d, total_ce_s1, total_ce_s2, total_ce_s3 = 0, 0, 0, 0
        total_kl_r1, total_kl_r2, total_kl_r3 = 0, 0, 0
        total_mse_1, total_mse_2, total_mse_3 = 0, 0, 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data, gt_label) in enumerate(t):
                # print('taxonomy_ids:', taxonomy_ids)
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']
                # print('train:', partial.shape, gt.shape)
                gt_label = utils.helpers.var_or_cuda(torch.Tensor(gt_label).long())

                pcds_pred, labels_pred, feats_cls = model(partial)
                # print('train in and pred and gt:', partial.shape, pcds_pred[-1].shape, gt.shape)

                loss_total, losses, cur_idx, best_idx = get_loss_nomi(labels_pred, gt_label, pcds_pred, partial, 
                                                            gt, feats_cls, last_idx, indices, epoch_idx,
                                                            mse=cfg.TRAIN.CODE, nomi=cfg.TRAIN.NOMI)
                last_idx = best_idx

                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.95)
                optimizer.step()

                accs = []
                for logit in labels_pred:
                    pred = logit.argmax(-1)
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
                if cfg.TRAIN.CODE:
                    mse_1_item = losses[10].item()# * 1e2
                    total_mse_1 += mse_1_item
                    mse_2_item = losses[11].item()# * 1e2
                    total_mse_2 += mse_2_item
                    mse_3_item = losses[12].item()# * 1e2
                    total_mse_3 += mse_3_item
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description('[Epoch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS))
                t.set_postfix(cd='%s' % ['%.4f' % l for l in [cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item]],
                              acc='%s' % ['%.4f' % acc for acc in accs],
                              kl='%s' % ['%.4f' % kl for kl in [kl_r1_item, kl_r2_item]])

                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

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
            train_writer.add_scalar('Loss/Epoch/total_mse_1', avg_mse1, epoch_idx)
            avg_mse2 = total_mse_2 / n_batches
            avg_mse3 = total_mse_3 / n_batches
            train_writer.add_scalar('Loss/Epoch/total_mse_2', avg_mse2, epoch_idx)
            train_writer.add_scalar('Loss/Epoch/total_mse_3', avg_mse3, epoch_idx)

        lr_scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'], 'best:', best_idx)
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p3', avg_cd3, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/partial_matching', avg_partial, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/ce_s1', avg_ce1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/ce_s2', avg_ce2, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/ce_s3', avg_ce3, epoch_idx)
        train_writer.add_scalar('Hyper/Epoch/best_idx', best_idx, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/kl_r1', avg_kl1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/kl_r2', avg_kl2, epoch_idx)
        train_writer.add_scalar('Hyper/Epoch/lr', optimizer.param_groups[0]['lr'], epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s Losses_CE = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]],
             ['%.4f' % l for l in [avg_ce1, avg_ce2, avg_ce3]]))

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
        if max(best_acc) > max(best_metrics) and epoch_idx > 50:
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, 'ckpt-best.pth')
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_acc,
                'best_metrics_mean': best_mean,
                'best_metrics_CD': best_metrics_CD,
                'scheduler': lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict()
            }, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

        if best_acc[0] > best_metrics[0]:
            best_metrics[0] = best_acc[0]
        if best_acc[1] > best_metrics[1]:
            best_metrics[1] = best_acc[1]
        if best_acc[2] > best_metrics[2]:
            best_metrics[2] = best_acc[2]
        if best_mean[0] > best_metrics_mean[0]:
            best_metrics_mean[0] = best_mean[0]
        if best_mean[1] > best_metrics_mean[1]:
            best_metrics_mean[1] = best_mean[1]
        if best_mean[2] > best_metrics_mean[2]:
            best_metrics_mean[2] = best_mean[2]
        if best_CD < best_metrics_CD:
            best_metrics_CD = best_CD
        print(f'Best acc: {best_metrics}, mean: {best_metrics_mean}, CD: {best_metrics_CD}')
        # log_file.write(f'[{epoch_idx}] Best acc: {best_metrics}, mean: {best_metrics_mean}, CD: {best_metrics_CD}\n')

    train_writer.close()
    val_writer.close()
    # log_file.close()
