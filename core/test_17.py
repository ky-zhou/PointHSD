import logging
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.loss_utils import *
from models.curvenet_cls import CurveNet_SD as Model
# from models.model17 import PointNet as Model
# from models.model17 import PointMLP as Model
import numpy as np
from core.visualize import *
import sklearn.metrics as metrics


def calculate_entropy(feature_values):
    unique_values, value_counts = np.unique(feature_values, return_counts=True)
    probabilities = value_counts / len(feature_values)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None, show=False, save_path=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        test_data_loader = torch.utils.data.DataLoader(utils.data_loaders.ModelNet40H5(partition='test', num_points=1024), num_workers=cfg.CONST.NUM_WORKERS//2,
                                 batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last=False)

    # Setup networks and initialize networks
    if model is None:
        model = Model(40)
        # model = Model(points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
        #            activation="relu", bias=False, use_xyz=False, normalize="anchor",
        #            dim_expansion=[2, 2, 2], pre_blocks=[2, 2, 2], pos_blocks=[2, 2, 2],
        #            k_neighbors=[16, 32, 48], reducers=[2, 2, 2])
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        save_path = cfg.CONST.WEIGHTS.replace('checkpoints', 'figs').rstrip('/ckpt-best.pth')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Switch models to evaluation mode
    model.eval()

    test_accs, mAccs, entropies = [], [], [] # for all categories
    test_trues = {0: [], 1: [], 2:[]}
    test_preds = {0: [], 1: [], 2:[]}
    test_feats = {0: [], 1: [], 2:[]}
    # Testing loop
    with tqdm(test_data_loader) as t: # each batch seems to have only one taxonomy, at least in test set
        for model_idx, (data, gt_label) in enumerate(t):

            with torch.no_grad():
                data = utils.helpers.var_or_cuda(data)
                # print('train:', partial.shape, gt.shape)
                gt_label = utils.helpers.var_or_cuda(gt_label.squeeze())

                labels_pred, features = model(data.permute(0, 2, 1).contiguous())

                for i, logit in enumerate(labels_pred):
                    pred = logit.argmax(-1)
                    test_trues[i].append(gt_label.cpu().detach().numpy())
                    test_preds[i].append(pred.cpu().detach().numpy())
                    test_feats[i].append(features[i].cpu().detach().numpy())

    for j in range(3):
        test_true = np.concatenate(test_trues[j])
        test_pred = np.concatenate(test_preds[j])
        test_feat = np.concatenate(test_feats[j])
        test_accs.append(metrics.accuracy_score(test_true, test_pred))
        mAccs.append(metrics.balanced_accuracy_score(test_true, test_pred))
        entropies.append(calculate_entropy(test_feat))


    print(f'Overall: {test_accs}, mean: {mAccs}, entropy: {entropies}')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/Acc1', test_accs[0], epoch_idx)
        test_writer.add_scalar('Loss/Epoch/Acc2', test_accs[1], epoch_idx)
        test_writer.add_scalar('Loss/Epoch/Acc3', test_accs[2], epoch_idx)
        test_writer.add_scalar('Loss/Epoch/mAcc1', mAccs[0], epoch_idx)
        test_writer.add_scalar('Loss/Epoch/mAcc2', mAccs[1], epoch_idx)
        test_writer.add_scalar('Loss/Epoch/mAcc3', mAccs[2], epoch_idx)

    return test_accs, mAccs
