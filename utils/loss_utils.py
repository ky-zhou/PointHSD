import torch
import collections
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.utils import fps_subsample
chamfer_dist = chamfer_3DDist()


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, idx1, idx2 = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2, idx1, idx2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, idx1, idx2 = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1, idx1, idx2


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, idx1, idx2 = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1, idx1, idx2


def get_loss_up_1ce(pcds_pred, labels_pred, partial, gt, gt_label, feats_cls, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc, _, _  = CD(Pc, gt_c)
    cd1, _, _  = CD(P1, gt_1)
    cd2, _, _  = CD(P2, gt_2)
    cd3, _, _  = CD(P3, gt)

    partial_matching, _, _  = PM(partial, P3)

    CE = torch.nn.CrossEntropyLoss()
    ced = CE(labels_pred, gt_label)

    loss_cd = cdc + cd1 + cd2 + cd3 + partial_matching
    loss_ce = ced

    loss_all = loss_cd * 1e3 + loss_ce# + beta * loss_l2
    losses = [cdc, cd1, cd2, cd3, partial_matching, ced]
    return loss_all, losses


def get_loss_wo_bcs(pcds_pred, labels_pred, partial, gt, gt_label, feats_cls, sqrt=True, mse=False): # without best idx
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt, P1.shape[1])
    gt_c = fps_subsample(gt, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, gt)

    partial_matching = PM(partial, P3)

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred[0], gt_label)
    ce2 = CE(labels_pred[1], gt_label)
    ce3 = CE(labels_pred[2], gt_label)

    # use last classifier as reference label
    KL = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    label_pred_soft0 = torch.nn.functional.log_softmax(labels_pred[0], dim=-1)
    label_pred_soft1 = torch.nn.functional.log_softmax(labels_pred[1], dim=-1)
    label_pred_soft2 = torch.nn.functional.log_softmax(labels_pred[2], dim=-1)

    kl1 = KL(label_pred_soft0, label_pred_soft2)
    kl2 = KL(label_pred_soft1, label_pred_soft2)

    # L2 feature loss
    MSE = torch.nn.MSELoss()
    mse1 = MSE(feats_cls[0], feats_cls[2])
    mse2 = MSE(feats_cls[1], feats_cls[2])

    alpha, beta = 0.2, 0.5

    loss_cd = cdc + cd1 + cd2 + cd3 + partial_matching
    loss_ce = ce1 + ce2 + ce3
    loss_kl = kl1 + kl2
    loss_l2 = mse1 + mse2 if mse else 0
    loss_dis = (1 - alpha) * loss_ce + alpha * loss_kl
    loss_all = loss_cd * 1e3 + loss_dis + beta * loss_l2
    losses = [cdc, cd1, cd2, cd3, partial_matching, ce1, ce2, ce3, kl1, kl2]
    return loss_all, losses


def get_loss_nomi(labels_pred, gt_label, pcds_pred, partial, gt, feats_cls, last_idx, indices,
                  epoch_idx, sqrt=True, mse=False, nomi=False): # without best idx
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    # nomi_teacher = True if last_batch else False
    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc, _, _ = CD(Pc, gt_c)
    cd1, idx1, idx2 = CD(P1, gt_1)
    cd2, idx1, idx2 = CD(P2, gt_2)
    cd3, idx1, idx2 = CD(P3, gt)

    partial_matching, _, _ = PM(partial, P3)

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred[0], gt_label)
    ce2 = CE(labels_pred[1], gt_label)
    ce3 = CE(labels_pred[2], gt_label)
    # print(f'loss label and gt_label after gathering: {labels_pred[0].shape, gt_label.shape}')
    # ced = CE(labels_pred[3], gt_label)
    cur_idx = torch.argmin(torch.tensor([ce1, ce2, ce3], requires_grad=True))
    indices.append(cur_idx.item())
    if nomi:
        # counter = collections.Counter(indices)
        # best_idx = counter.most_common(1)[0][0]
        best_idx = 0 if epoch_idx < 100 else 2
    else:
        best_idx = last_idx
    # indices.append(cur_idx.item())
    # counter = collections.Counter(indices)
    # best_idx = counter.most_common(1)[0][0]
    # use last classifier as reference label

    KL = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    label_pred_soft0 = torch.nn.functional.log_softmax(labels_pred[0], dim=-1)
    label_pred_soft1 = torch.nn.functional.log_softmax(labels_pred[1], dim=-1)
    label_pred_soft2 = torch.nn.functional.log_softmax(labels_pred[2], dim=-1)
    label_pred_softs = [label_pred_soft0, label_pred_soft1, label_pred_soft2]
    loss_kl, kls = 0, []
    if nomi:
        for s in range(len(label_pred_softs)):
            if s != best_idx:
                kl = KL(label_pred_softs[s], label_pred_softs[best_idx])
                loss_kl += kl
                kls.append(kl)
        kl1, kl2 = kls
    else: # the deepest is the teacher
        kl1 = KL(label_pred_soft0, label_pred_soft2)
        kl2 = KL(label_pred_soft1, label_pred_soft2)
        loss_kl = kl1 + kl2


    # L2 feature loss
    # MSE = torch.nn.MSELoss()
    # if mse:
    #     mse1 = torch.tensor(0)
    #     mse2 = MSE(feats_cls[1], feats_cls[3])
    #     mse3 = MSE(feats_cls[2], feats_cls[3])
    # else:
    #     mse1 = torch.tensor(0)
    #     mse2 = torch.tensor(0)
    #     mse3 = torch.tensor(0)
    # kl1, kl2 = torch.tensor(0), torch.tensor(0)
    mse1, mse2, mse3= torch.tensor(0), torch.tensor(0), torch.tensor(0)
    alpha, beta, theta = 0.2, 1.0, 10.0


    loss_cd = cdc + cd1 + cd2 + cd3 + partial_matching
    loss_ce = ce1 + ce2 + ce3
    # loss_kl = kl1 + kl2
    # loss_l2 = mse1 if epoch_idx < 100 else  mse2 + mse3 if mse else 0
    loss_l2 = mse2 + mse3 if mse else 0
    loss_dis = (1 - alpha) * loss_ce + alpha * loss_kl
    loss_all = loss_cd * 1e3 + loss_dis + beta * loss_l2 #+ theta * loss_sdf
    # loss_all = loss_cd * 1e3 + loss_ce
    losses = [cdc, cd1, cd2, cd3, partial_matching, ce1, ce2, ce3, kl1, kl2, mse1, mse2, mse3]
    return loss_all, losses, cur_idx, best_idx


def get_loss_9(labels_pred, gt_label, pcds_pred, partial, gt, infos, sqrt=True): # without best idx
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    # nomi_teacher = True if last_batch else False
    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc, _, _ = CD(Pc, gt_c)
    cd1, idx11, idx12 = CD(P1, gt_1)
    cd2, idx21, idx22 = CD(P2, gt_2)
    cd3, idx31, idx32 = CD(P3, gt)
    partial_matching, _, _ = PM(partial, P3)
    # labels_pred[0] = labels_pred[0].argmax(1).gather(1, idx1.type(torch.int64))
    # labels_pred[1] = labels_pred[1].argmax(1).gather(1, idx1.type(torch.int64))
    # labels_pred[2] = labels_pred[2].argmax(1).gather(1, idx1.type(torch.int64))
    # label_pred_logit0 = torch.gather(labels_pred[0].argmax(1), 1, idx1.type(torch.int64))
    # label_pred_logit1 = labels_pred[1].argmax(1).gather(1, idx1.type(torch.int64))
    # label_pred_logit2 = labels_pred[2].argmax(1).gather(1, idx1.type(torch.int64))
    gt_label0 = gt_label.gather(1, idx31.type(torch.int64)) # gt uses idx1 to convert to pred's index
    # gt_label2 = gt_label.gather(1, idx22.type(torch.int64))
    # gt_label3 = gt_label.gather(1, idx23.type(torch.int64))

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred[0], gt_label0)
    ce2 = CE(labels_pred[1], gt_label0)
    ce3 = CE(labels_pred[2], gt_label0)

    label_pred_soft0 = torch.nn.functional.log_softmax(infos[0], dim=1)
    label_pred_soft1 = torch.nn.functional.log_softmax(infos[1], dim=1)
    label_pred_soft2 = torch.nn.functional.log_softmax(infos[2], dim=1)
    # idx32 = idx32.unsqueeze(1).repeat(1, label_pred_soft0.shape[1], 1)
    # label_pred_soft0 = label_pred_soft0.gather(2, idx32.type(torch.int64))
    # label_pred_soft1 = label_pred_soft1.gather(2, idx32.type(torch.int64))
    # label_pred_soft2 = label_pred_soft2.gather(2, idx32.type(torch.int64)) 
    # print(idx32.shape)
    # print(label_pred_soft0.shape, label_pred_soft1.shape, label_pred_soft2.shape)
    KL = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    label_pred_softs = [label_pred_soft0, label_pred_soft1, label_pred_soft2]
    loss_kl, kls = 0, []
    kl1 = KL(label_pred_soft0, label_pred_soft2)
    kl2 = KL(label_pred_soft1, label_pred_soft2)
    loss_kl = kl1 + kl2



    # kl1, kl2 = torch.tensor(0), torch.tensor(0)
    mse1, mse2, mse3= torch.tensor(0), torch.tensor(0), torch.tensor(0)
    alpha, beta, theta = 0.2, 1.0, 10.0


    loss_cd = cdc + cd1 + cd2 + cd3 + partial_matching
    loss_ce = ce1 + ce2 + ce3
    # loss_kl = kl1 + kl2
    # loss_l2 = mse1 if epoch_idx < 100 else  mse2 + mse3 if mse else 0
    loss_dis = (1 - alpha) * loss_ce + alpha * loss_kl
    loss_all = loss_cd * 1e3 + loss_dis #+ theta * loss_sdf
    # loss_all = loss_cd * 1e3 + loss_ce
    losses = [cdc, cd1, cd2, cd3, partial_matching, ce1, ce2, ce3, kl1, kl2, mse1, mse2, mse3]
    return loss_all, losses, (idx11, idx21, idx31), (idx12, idx22, idx32)


def get_loss_10(labels_pred, gt_label, pcds_pred, partial, gt, sqrt=True): # without best idx
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    # nomi_teacher = True if last_batch else False
    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc, _, _ = CD(Pc, gt_c)
    cd1, idx11, idx12 = CD(P1, gt_1)
    cd2, idx21, idx22 = CD(P2, gt_2)
    cd3, idx31, idx32 = CD(P3, gt)
    partial_matching, _, _ = PM(partial, P3)
    # labels_pred0 = labels_pred[0].gather(-1, idx31.type(torch.int64))
    # labels_pred[1] = labels_pred[1].argmax(1).gather(1, idx1.type(torch.int64))
    gt_label0 = gt_label.gather(1, idx31.type(torch.int64)) # gt uses idx1 to convert to pred's index
    # print(f'loss label and gt_label after gathering: {labels_pred[0].shape, gt_label.shape}')

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred[0], gt_label0)

    # kl1, kl2 = torch.tensor(0), torch.tensor(0)
    mse1, mse2, mse3= torch.tensor(0), torch.tensor(0), torch.tensor(0)
    alpha, beta, theta = 0.2, 1.0, 10.0


    loss_cd = cdc + cd1 + cd2 + cd3 + partial_matching
    loss_ce = ce1
    loss_all = loss_cd * 1e3 + loss_ce #+ theta * loss_sdf
    # loss_all = loss_cd * 1e3 + loss_ce
    losses = [cdc, cd1, cd2, cd3, partial_matching, ce1, mse1, mse2, mse3]
    return loss_all, losses, (idx11, idx21, idx31), (idx12, idx22, idx32)


def get_loss_nomi23(labels_pred, gt_label, pcds_pred, partial, gt, feats_cls, last_idx, indices,
                   epoch_idx, sqrt=True, mse=False, nomi=False): # without best idx
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    # nomi_teacher = True if last_batch else False
    Pc, P1, P2 = pcds_pred

    # gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt, P1.shape[1])
    gt_c = fps_subsample(gt, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt)

    # cdc = CD(Pc, gt)
    # cd1 = CD(P1, gt)
    # cd2 = CD(P2, gt)
    # cd3 = CD(P3, gt)

    partial_matching = PM(partial, P2)

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred[0], gt_label)
    ce2 = CE(labels_pred[1], gt_label)
    cur_idx = torch.argmin(torch.tensor([ce1, ce2], requires_grad=True))
    # print(cur_idx)
    indices.append(cur_idx.item())
    if nomi:
        # counter = collections.Counter(indices)
        # best_idx = counter.most_common(1)[0][0]
        best_idx = 0 if epoch_idx < 100 else 1

    else:
        best_idx = last_idx
    # indices.append(cur_idx.item())
    # counter = collections.Counter(indices)
    # best_idx = counter.most_common(1)[0][0]
    # use last classifier as reference label

    # KL = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    # label_pred_soft0 = torch.nn.functional.log_softmax(labels_pred[0], dim=-1)
    # label_pred_soft1 = torch.nn.functional.log_softmax(labels_pred[1], dim=-1)
    # label_pred_softs = [label_pred_soft0, label_pred_soft1]
    # loss_kl, kls = 0, []
    # if nomi:
    #     for s in range(len(label_pred_softs)):
    #         if s != best_idx:
    #             kl = KL(label_pred_softs[s], label_pred_softs[best_idx])
    #             loss_kl += kl
    #             kls.append(kl)
    #     kl1 = kls[0]
    # else: # the deepest is the teacher
    #     kl1 = KL(label_pred_soft0, label_pred_soft1)
    #     loss_kl = kl1


    # L2 feature loss
    # MSE = torch.nn.MSELoss()
    # if mse:
    #     mse1 = torch.tensor(0)
    #     mse2 = MSE(feats_cls[1], feats_cls[2])
    # else:
    #     mse1 = torch.tensor(0)
    #     mse2 = torch.tensor(0)
    kl1, mse1, mse2 = torch.tensor(0), torch.tensor(0), torch.tensor(0)
    alpha, beta, theta = 0.2, 1.0, 10.0


    loss_cd = cdc + cd1 + cd2 + partial_matching
    loss_ce = ce1 + ce2 
    loss_kl = kl1 
    # loss_l2 = mse2 if mse else 0
    # loss_l2 = mse1 if mse else 0
    # loss_dis = (1 - alpha) * loss_ce + alpha * loss_kl
    # loss_all = loss_cd * 1e3 + loss_dis# + beta * loss_l2 #+ theta * loss_sdf
    loss_all = loss_cd * 1e3 + loss_ce# + beta * loss_l2 #+ theta * loss_sdf
    losses = [cdc, cd1, cd2, partial_matching, ce1, ce2, kl1, mse1, mse2]
    return loss_all, losses, cur_idx, best_idx


def get_loss_nomi22(labels_pred, gt_label, pcds_pred, partial, gt, feats_cls, last_idx, indices,
                   sqrt=True, mse=False, nomi=False): # without best idx
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    # nomi_teacher = True if last_batch else False
    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt, P1.shape[1])
    gt_c = fps_subsample(gt, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, gt)

    partial_matching = PM(partial, P3)

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred[0], gt_label)
    ce2 = CE(labels_pred[1], gt_label)
    ce3 = CE(labels_pred[2], gt_label)
    ce4 = CE(labels_pred[3], gt_label)
    cur_idx = torch.argmin(torch.tensor([ce1, ce2, ce3, ce4], requires_grad=True))
    indices.append(cur_idx.item())
    if nomi:
        counter = collections.Counter(indices)
        best_idx = counter.most_common(1)[0][0]
    else:
        best_idx = last_idx
    # indices.append(cur_idx.item())
    # counter = collections.Counter(indices)
    # best_idx = counter.most_common(1)[0][0]
    # use last classifier as reference label
    KL = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    label_pred_soft0 = torch.nn.functional.log_softmax(labels_pred[0], dim=-1)
    label_pred_soft1 = torch.nn.functional.log_softmax(labels_pred[1], dim=-1)
    label_pred_soft2 = torch.nn.functional.log_softmax(labels_pred[2], dim=-1)
    label_pred_soft3 = torch.nn.functional.log_softmax(labels_pred[3], dim=-1)

    label_pred_softs = [label_pred_soft0, label_pred_soft1, label_pred_soft2, label_pred_soft3]
    loss_kl, kls = 0, []
    if nomi:
        for s in label_pred_softs:
            if s is not label_pred_softs[best_idx]:
                kl = KL(s, label_pred_softs[best_idx])
                loss_kl += kl
                kls.append(kl)
        kl1, kl2, kl3 = kls
    else: # the deepest is the teacher
        kl1 = KL(label_pred_soft0, label_pred_soft2)
        kl2 = KL(label_pred_soft1, label_pred_soft2)
        kl3 = KL(label_pred_soft3, label_pred_soft2)
        loss_kl = kl1 + kl2 + kl3


    # L2 feature loss
    MSE = torch.nn.MSELoss()
    # print(f'MSE: {feats_cls[0].shape, feats_cls[3].shape}')
    # mse1 = MSE(feats_cls[0], feats_cls[3]) # last is the code from snowflake
    # mse2 = MSE(feats_cls[1], feats_cls[3])
    # mse3 = MSE(feats_cls[2], feats_cls[3])
    # mse1 = MSE(feats_cls[3], feats_cls[2]) # last is the code from snowflake
    # mse2 = MSE(feats_cls[1], feats_cls[2])
    # mse3 = MSE(feats_cls[0], feats_cls[2])
    mse1 = MSE(feats_cls[1], feats_cls[0]) # gradual mse, reversed
    mse2 = MSE(feats_cls[2], feats_cls[0])
    mse3 = MSE(feats_cls[3], feats_cls[0])
    # mse1 = MSE(feats_cls[0], feats_cls[3]) # gradual mse
    # mse2 = MSE(feats_cls[1], feats_cls[0])
    # mse3 = MSE(feats_cls[2], feats_cls[1])
    # mse1 = MSE(feats_cls[3], feats_cls[1]) # use 1 as guidance
    # mse2 = MSE(feats_cls[0], feats_cls[1])
    # mse3 = MSE(feats_cls[2], feats_cls[1])

    # sdf1 = torch.linalg.norm((gt_1 - P1), ord=2, dim=-1).mean()
    # sdf2 = torch.linalg.norm((gt_2 - P2), ord=2, dim=-1).mean()
    # sdf3 = torch.linalg.norm((gt - P3), ord=2, dim=-1).mean()
    # loss_sdf = sdf1 + sdf2 + sdf3

    alpha, beta, theta = 0.2, 1.0, 10.0


    loss_cd = cdc + cd1 + cd2 + cd3 + partial_matching
    loss_ce = ce1 + ce2 + ce3
    loss_kl = kl1 + kl2 + kl3
    loss_l2 = mse1 + mse2 + mse3 if mse else 0
    loss_dis = (1 - alpha) * loss_ce + alpha * loss_kl
    loss_all = loss_cd * 1e3 + loss_dis + beta * loss_l2 #+ theta * loss_sdf
    losses = [cdc, cd1, cd2, cd3, partial_matching, ce1, ce2, ce3, ce4, kl1, kl2, kl3, mse1, mse2, mse3]
    return loss_all, losses, cur_idx, best_idx


def get_loss_1ce(labels_pred, gt_label): # only cross entropy
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred, gt_label)
    loss_all = ce1# + beta * loss_l2
    losses = [ce1]
    return loss_all, losses


def get_loss_ensem(labels_pred, gt_label): # only cross entropy
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred[0], gt_label)
    ce2 = CE(labels_pred[1], gt_label)
    ce3 = CE(labels_pred[2], gt_label)
    ce4 = CE(labels_pred[3], gt_label)

    # use last classifier as reference label
    KL = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    label_pred_soft0 = torch.nn.functional.log_softmax(labels_pred[0], dim=-1)
    label_pred_soft1 = torch.nn.functional.log_softmax(labels_pred[1], dim=-1)
    label_pred_soft2 = torch.nn.functional.log_softmax(labels_pred[2], dim=-1)
    label_pred_soft3 = torch.nn.functional.log_softmax(labels_pred[3], dim=-1)
    kl1 = KL(label_pred_soft0, label_pred_soft3)
    kl2 = KL(label_pred_soft1, label_pred_soft3)
    kl3 = KL(label_pred_soft2, label_pred_soft3)

    alpha, beta = 0.2, 0.5

    loss_ce = ce1 + ce2 + ce3 + ce4
    loss_kl = kl1 + kl2 + kl3
    # loss_l2 = mse1 + mse2
    loss_dis = (1 - alpha) * loss_ce + alpha * loss_kl
    loss_all = loss_dis# + beta * loss_l2
    losses = [ce1, ce2, ce3, ce4, kl1, kl2, kl3]
    return loss_all, losses


def get_loss_ensem_sd(labels_pred, gt_label, pcds_pred, partial, gt, feats_cls,
                  epoch_idx, sqrt=True, mse=False, nomi=False): # without best idx
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    # nomi_teacher = True if last_batch else False
    Pc, P1, P2, P3 = pcds_pred

    # gt_2 = fps_subsample(gt, P2.shape[1])
    # gt_1 = fps_subsample(gt, P1.shape[1])
    # gt_c = fps_subsample(gt, Pc.shape[1])

    # cdc = CD(Pc, gt_c)
    # cd1 = CD(P1, gt_1)
    # cd2 = CD(P2, gt_2)
    # cd3 = CD(P3, gt)

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, gt)


    partial_matching = PM(partial, P3)

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred[0], gt_label)
    ce2 = CE(labels_pred[1], gt_label)
    ce3 = CE(labels_pred[2], gt_label)
    ce4 = CE(labels_pred[3], gt_label)

    # use last classifier as reference label
    KL = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    label_pred_soft0 = torch.nn.functional.log_softmax(labels_pred[0], dim=-1)
    label_pred_soft1 = torch.nn.functional.log_softmax(labels_pred[1], dim=-1)
    label_pred_soft2 = torch.nn.functional.log_softmax(labels_pred[2], dim=-1)
    label_pred_soft3 = torch.nn.functional.log_softmax(labels_pred[3], dim=-1)
    kl1 = KL(label_pred_soft0, label_pred_soft3)
    kl2 = KL(label_pred_soft1, label_pred_soft3)
    kl3 = KL(label_pred_soft2, label_pred_soft3)

    MSE = torch.nn.MSELoss()
    if mse:
        mse1 = MSE(feats_cls[0], feats_cls[3]) # gradual mse, reversed
        mse2 = MSE(feats_cls[1], feats_cls[3])
        mse3 = MSE(feats_cls[2], feats_cls[3])
    else:
        mse1 = torch.tensor(0)
        mse2 = torch.tensor(0)
        mse3 = torch.tensor(0)

    alpha, beta, theta = 0.2, 1.0, 10.0

    loss_cd = cdc + cd1 + cd2 + cd3 + partial_matching
    loss_ce = ce1 + ce2 + ce3 + ce4
    loss_kl = kl1 + kl2
    loss_l2 = mse1 +  mse2 + mse3 if mse else 0
    loss_dis = (1 - alpha) * loss_ce + alpha * loss_kl
    loss_all = loss_cd * 1e3 + loss_dis + beta * loss_l2 #+ theta * loss_sdf
    losses = [cdc, cd1, cd2, cd3, partial_matching, ce1, ce2, ce3, ce4, kl1, kl2, kl3, mse1, mse2, mse3]
    return loss_all, losses


def get_loss_3ce(labels_pred, gt_label): # only cross entropy
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred[0], gt_label)
    ce2 = CE(labels_pred[1], gt_label)
    ce3 = CE(labels_pred[2], gt_label)

    # use last classifier as reference label
    KL = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    label_pred_soft0 = torch.nn.functional.log_softmax(labels_pred[0], dim=-1)
    label_pred_soft1 = torch.nn.functional.log_softmax(labels_pred[1], dim=-1)
    label_pred_soft2 = torch.nn.functional.log_softmax(labels_pred[2], dim=-1)
    kl1 = KL(label_pred_soft0, label_pred_soft2)
    kl2 = KL(label_pred_soft1, label_pred_soft2)


    alpha, beta = 0.2, 0.5

    loss_ce = ce1 + ce2 + ce3
    loss_kl = kl1 + kl2
    # loss_l2 = mse1 + mse2
    loss_dis = (1 - alpha) * loss_ce + alpha * loss_kl
    loss_all = loss_dis# + beta * loss_l2
    losses = [ce1, ce2, ce3, kl1, kl2]
    return loss_all, losses


def get_loss_3ce_nomi(labels_pred, gt_label, feats_cls, last_idx, indices, mse=False, nomi=False):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """

    CE = torch.nn.CrossEntropyLoss()
    ce1 = CE(labels_pred[0], gt_label)
    ce2 = CE(labels_pred[1], gt_label)
    ce3 = CE(labels_pred[2], gt_label)

    # use last classifier as reference label
    KL = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    label_pred_soft0 = torch.nn.functional.log_softmax(labels_pred[0], dim=-1)
    label_pred_soft1 = torch.nn.functional.log_softmax(labels_pred[1], dim=-1)
    label_pred_soft2 = torch.nn.functional.log_softmax(labels_pred[2], dim=-1)

    cur_idx = torch.argmin(torch.tensor([ce1, ce2, ce3], requires_grad=True))
    indices.append(cur_idx.item())
    # if nomi:
    counter = collections.Counter(indices)
    best_idx = counter.most_common(1)[0][0]
        # print(best_idx[0][0], counter)
    # else:
    #     best_idx = last_idx

    label_pred_softs = [label_pred_soft0, label_pred_soft1, label_pred_soft2]
    loss_kl, kls = 0, []
    for s in label_pred_softs:
        if s is not label_pred_softs[best_idx]:
            kl = KL(s, label_pred_softs[best_idx])
            loss_kl += kl
            kls.append(kl)

    kl1, kl2 = kls
    # L2 feature loss
    if mse:
        MSE = torch.nn.MSELoss()
        mse1 = MSE(feats_cls[0], feats_cls[2])
        mse2 = MSE(feats_cls[1], feats_cls[2])

    alpha, beta = 0.2, 0.5

    loss_ce = ce1 + ce2 + ce3
    loss_kl = kl1 + kl2
    loss_l2 = mse1 + mse2 if mse else 0
    loss_dis = (1 - alpha) * loss_ce + alpha * loss_kl
    loss_all = loss_dis + beta * loss_l2
    losses = [ce1, ce2, ce3, kl1, kl2]
    return loss_all, losses, cur_idx, best_idx

