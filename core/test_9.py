import logging
import torch
import utils.data_loaders as dload
import utils.helpers
from tqdm import tqdm
from utils.metrics import compute_overall_iou
from utils.loss_utils import *
from models.model9 import SnowflakeNet as Model
import numpy as np
from core.visualize import *
import json


num_part = 50


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None, show=False, save_path=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = dload.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(dload.DatasetSubset.TEST),
                                                       batch_size=100,
                                                       num_workers=cfg.CONST.NUM_WORKERS//2,
                                                       collate_fn=dload.collate_fn4,
                                                       pin_memory=True,
                                                       shuffle=False)
        color_json = './datasets/shapenet_part_seg_hdf5_data/part_color_mapping.json'
        with open(color_json) as f:
            part_colors = json.loads(f.read()) # 50*3

    # Setup networks and initialize networks
    if model is None:
        model = Model(dim_feat=512, up_factors=[1, 2], dim_cat=num_part)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        record = checkpoint['per_cat_iou']
        print(record)
        best_miou, best_ciou = checkpoint['best_metrics'], checkpoint['best_metrics_cls']
        print(best_miou, best_ciou)
        save_path = cfg.CONST.WEIGHTS.replace('checkpoints', 'figs').rstrip('/ckpt-best.pth')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Switch models to evaluation mode
    model.eval()

    n_batches = len(test_data_loader)
    pred_labels, gt_labels = [], []
    total_cd_p0, total_cd_p1, total_cd_p2, total_cd_p3, total_partial = 0, 0, 0, 0, 0
    total_ce_d, total_ce_s1, total_ce_s2, total_ce_s3 = 0, 0, 0, 0

    mious = [[], [], []]
    per_cat_iou = np.zeros((3, 16)).astype(np.float32) # cannot use [np(a)]]*m format, as np(a) will be duplicated for m times
    per_cat_seen = np.zeros((3, 16)).astype(np.int32)
    avg_iou = [0.0] * 3 # though no bug for this form, its strongly recommended not to use this form, here only for speed
    with tqdm(test_data_loader) as t: # each batch seems to have only one taxonomy, at least in test set
        for batch_idx, (taxonomy_id, model_id, data, data_label, data_cls_label) in enumerate(t):
            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                for k, v in data_label.items():
                    data_label[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']
                gt_label = data_label['gtlabel'].cuda().long()
                # data_cls_label = data_cls_label.cuda().long()
                # print(f'test partial {partial.shape}')
                # print(f'test gt_label {gt_label.shape}') # 64

                batch_size, n, _ = gt.shape
                # print('batch_size:', batch_size)

                pcds_pred, labels_pred, infos = model(partial.contiguous(), dload.to_categorical(data_cls_label, num_classes=16))
                # print(f'partial {partial.shape}')
                # print(f'taxonomy_id {taxonomy_id, type(taxonomy_id)}')
                # print(f'model_id {model_id, type(taxonomy_id)}')
                # loss_total, losses = get_loss(pcds_pred, partial, gt, sqrt=True)
                loss_total, losses, idx1s, idx2s = get_loss_9(labels_pred, gt_label, pcds_pred, partial, gt, infos, sqrt=True)
                # print(f'test cd_idx1: {cd_idx1}')

                # category count
                cd_p0_item = losses[0].item() * 1e3
                total_cd_p0 += cd_p0_item
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

                """PointMLP IOU implementation"""
                gt_label_idx1 = gt_label.gather(1, idx1s[-1].type(torch.int64))
                for level in range(len(labels_pred)):
                    # assert idx1s[0] == idx1s[1]
                    batch_ious = compute_overall_iou(labels_pred[level], gt_label_idx1, num_part)  # [b]

                    for shape_idx in range(batch_size):  # sample_idx
                        cur_gt_label = data_cls_label[shape_idx]  # label[sample_idx], denotes current sample belongs to which cat
                        per_cat_iou[level, cur_gt_label] += batch_ious[shape_idx]  # add the iou belongs to this cat
                        per_cat_seen[level, cur_gt_label] += 1  # count the number of this cat is chosen
                    
                    mious[level].append(batch_ious)

                if show and epoch_idx % cfg.TRAIN.SAVE_FREQ == 0:
                    visual_data = {}
                    visual_data['pred'] = pcds_pred[-1].cpu()
                    visual_data['part'] = partial.cpu()
                    visual_data['original'] = gt.cpu()
                    plot_fig_seg(visual_data, save_path, batch_idx, labels_pred[-1].cpu().numpy(), gt_label.cpu().numpy())

                # save for visualization
                # method_name = 'pointmlp-hsd-part-s64-k16'
                # logit = labels_pred[-1].argmax(1).cpu().numpy().astype(np.int32)
                # gt_label_batch = gt_label.cpu().numpy().astype(np.int32)
                # for idx in range(len(model_id)):
                #     cat_dir = os.path.join(cfg.DIR.OUT_PATH, 'data-%s/%s' % (method_name, model_id[idx]))  # only used for testing
                #     if not os.path.exists(cat_dir):
                #         os.makedirs(cat_dir)
                #     """Save to xyz or npy"""
                #     colors_pred = np.array(list(map(lambda x: part_colors[x], logit[idx])))
                #     colors_gt = np.array(list(map(lambda x: part_colors[x], gt_label_batch[idx])))
                #     # file_path = os.path.join(cat_dir, '%s_%d_%d.ply' % (model_id[idx], batch_idx, idx))
                #     # pcd = o3d.geometry.PointCloud()
                #     # pcd.points = o3d.utility.Vector3dVector(pcds_pred[-1].data.cpu().numpy()[idx, ...])
                #     # pcd.colors = o3d.utility.Vector3dVector(colors_pred)
                #     # o3d.io.write_point_cloud(file_path, pcd)
                #     # file_path = os.path.join(cat_dir, '%s_%d_%d_gt.ply' % (model_id[idx], batch_idx, idx))
                #     # pcd = o3d.geometry.PointCloud()
                #     # pcd.points = o3d.utility.Vector3dVector(gt.data.cpu().numpy()[idx, ...])
                #     # pcd.colors = o3d.utility.Vector3dVector(colors_gt)
                #     # o3d.io.write_point_cloud(file_path, pcd)
                #     """Save to npy"""
                #     file_path = os.path.join(cat_dir, '%s_%d_%d_in.npy' % (model_id[idx], batch_idx, idx))
                #     np.save(file_path, partial.data.cpu().numpy()[idx, ...])
                #     file_path = os.path.join(cat_dir, '%s_%d_%d_pred.npy' % (model_id[idx], batch_idx, idx))
                #     data = np.concatenate((pcds_pred[-1].data.cpu().numpy()[idx, ...], colors_pred), 1)
                #     np.save(file_path, data)
                #     file_path = os.path.join(cat_dir, '%s_%d_%d_gt.npy' % (model_id[idx], batch_idx, idx))
                #     data = np.concatenate((gt.data.cpu().numpy()[idx, ...], colors_gt), 1)
                #     np.save(file_path, data)

    avg_cd0 = total_cd_p0 / n_batches
    avg_cd1 = total_cd_p1 / n_batches
    avg_cd2 = total_cd_p2 / n_batches
    avg_cd3 = total_cd_p3 / n_batches
    avg_partial = total_partial / n_batches
    avg_ce1 = total_ce_s1 / n_batches
    avg_ce2 = total_ce_s2 / n_batches
    avg_ce3 = total_ce_s3 / n_batches

    cls_iou = [0.0] * 3 #careful of using [object] * m, object could be duplicated
    for level in range(len(labels_pred)):
        for cat_idx in range(16):
            if per_cat_seen[level, cat_idx] > 0:  # indicating this cat is included during previous iou appending
                per_cat_iou[level, cat_idx] = per_cat_iou[level, cat_idx] / per_cat_seen[level, cat_idx]  # avg class iou across all samples
                # print(level, per_cat_iou[level, cat_idx])
                cls_iou[level] += per_cat_iou[level, cat_idx]
                # print(level, cls_iou[level])

        # metrics['accuracy'] = np.mean(accuracy)
        avg_iou[level] = np.mean(np.concatenate(mious[level]))
        cls_iou[level] /= 16
        # print('========', level, cls_iou[level])
    print('mIOU: {}, cls_IOU: {}, CD: {}'.format(avg_iou, cls_iou, avg_cd3))

    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd_p0', avg_cd0, epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd_p3', avg_cd3, epoch_idx)
        test_writer.add_scalar('Loss/Epoch/partial_matching', avg_partial, epoch_idx)
        test_writer.add_scalar('Loss/Epoch/ce_s1', avg_ce1, epoch_idx)
        test_writer.add_scalar('Loss/Epoch/ce_s2', avg_ce2, epoch_idx)
        test_writer.add_scalar('Loss/Epoch/ce_s3', avg_ce3, epoch_idx)
        test_writer.add_scalar('Metric/mIOU1', avg_iou[0], epoch_idx)
        test_writer.add_scalar('Metric/mIOU2', avg_iou[1], epoch_idx)
        test_writer.add_scalar('Metric/mIOU3', avg_iou[2], epoch_idx)
        test_writer.add_scalar('Metric/clsIOU1', cls_iou[0], epoch_idx)
        test_writer.add_scalar('Metric/clsIOU2', cls_iou[1], epoch_idx)
        test_writer.add_scalar('Metric/clsIOU3', cls_iou[2], epoch_idx)
    else:
        print(per_cat_iou)
    return avg_iou, cls_iou, per_cat_iou, avg_cd3