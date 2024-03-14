import torch
import torch.nn as nn
from models.utils import *
from models.skip_transformer import SkipTransformer
from models.pointmlp import index_points, knn_point
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
import torch.nn.functional as F
from .curvenet_util import *


class FCLayer(nn.Module):
    def __init__(self, in_channel, class_num) :
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel//2),
            nn.BatchNorm1d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_channel//2, in_channel//4),
            nn.BatchNorm1d(in_channel//4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(in_channel//4, class_num)
        )

    def forward(self, code):
        return self.fc(code)


class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class PointNet(nn.Module):
    def __init__(self, class_num):
        print('Downstream = PointNet')
        super(PointNet, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 12, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_21 = PointNet_SA_Module_KNN(128, 8, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_31 = PointNet_SA_Module_KNN(None, None, 256, [512, 512], group_all=True, if_bn=False)

        self.fc1 = FCLayer(512, class_num)

    def forward(self, point_cloud):
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l21_xyz, l21_points, idx21 = self.sa_module_21(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l31_xyz, l31_points = self.sa_module_31(l21_xyz, l21_points)  # (B, 3, 1), (B, out_dim, 1)
        logit1 = self.fc1(l31_points.squeeze(2))
        return logit1, [l21_points]


class PointNet_SD(nn.Module):
    def __init__(self, class_num):
        print('Downstream = PointNet-SD')
        super(PointNet_SD, self).__init__()
        self.sa_module_11 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_12 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_13 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_21 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_22 = PointNet_SA_Module_KNN(128, 32, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_23 = PointNet_SA_Module_KNN(128, 48, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_31 = PointNet_SA_Module_KNN(None, None, 256, [512, 512], group_all=True, if_bn=False)
        self.sa_module_32 = PointNet_SA_Module_KNN(None, None, 256, [512, 512], group_all=True, if_bn=False)
        self.sa_module_33 = PointNet_SA_Module_KNN(None, None, 256, [512, 512], group_all=True, if_bn=False)

        self.fc1 = FCLayer(512, class_num)
        self.fc2 = FCLayer(512, class_num)
        self.fc3 = FCLayer(512, class_num)

    def forward(self, point_cloud):
        l0_xyz = point_cloud
        l0_points = point_cloud
        # print(f'PointNet point_cloud {point_cloud.shape}')

        l11_xyz, l11_points, idx1 = self.sa_module_11(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l12_xyz, l12_points, idx1 = self.sa_module_12(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l13_xyz, l13_points, idx1 = self.sa_module_13(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        # print(f'PointNet 1 {l1_points.shape}')
        l21_xyz, l21_points, idx21 = self.sa_module_21(l11_xyz, l11_points)  # (B, 3, 128), (B, 256, 512)
        l22_xyz, l22_points, idx22 = self.sa_module_22(l12_xyz, l12_points)  # (B, 3, 128), (B, 256, 512)
        l23_xyz, l23_points, idx23 = self.sa_module_23(l13_xyz, l13_points)  # (B, 3, 128), (B, 256, 512)
        # print(f'PointNet 2 {l2_points.shape}')
        l31_xyz, l31_points = self.sa_module_31(l21_xyz, l21_points)  # (B, 3, 1), (B, out_dim, 1)
        l32_xyz, l32_points = self.sa_module_32(l22_xyz, l22_points)  # (B, 3, 1), (B, out_dim, 1)
        l33_xyz, l33_points = self.sa_module_33(l23_xyz, l23_points)  # (B, 3, 1), (B, out_dim, 1)
        # print(f'PointNet 3 {l3_points.shape}')
        l31_points = l31_points.squeeze(2)
        l32_points = l32_points.squeeze(2)
        l33_points = l33_points.squeeze(2)
        logit1 = self.fc1(l31_points)
        logit2 = self.fc2(l32_points)
        logit3 = self.fc3(l33_points)
        return [logit1, logit2, logit3], [l31_points, l32_points, l33_points]


class PointNet_SD_Cascade(nn.Module): # classification-only model
    def __init__(self, class_num):
        print('Downstream = PointNet-SD-Cascade')
        super(PointNet_SD_Cascade, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 12, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_12 = PointNet_SA_Module_KNN(128, 8, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_13 = PointNet_SA_Module_KNN(None, None, 256, [256, 512], group_all=True, if_bn=False)

        self.sa_module_22 = PointNet_SA_Module_KNN(256, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_23 = PointNet_SA_Module_KNN(None, None, 256, [512, 512], group_all=True, if_bn=False)

        self.sa_module_32 = PointNet_SA_Module_KNN(256, 24, 256, [256, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_33 = PointNet_SA_Module_KNN(None, None, 256, [512, 1024], group_all=True, if_bn=False)

        self.fc1 = FCLayer(512, class_num)
        self.fc2 = FCLayer(512, class_num)
        self.fc3 = FCLayer(1024, class_num)
        self.cls1 = nn.Linear(128, class_num)
        self.cls2 = nn.Linear(128, class_num)
        self.cls3 = nn.Linear(256, class_num)


    def forward(self, point_cloud):
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, _ = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l12_xyz, l12_points, _ = self.sa_module_12(l1_xyz, l1_points)  # (B, 3, 256), (B, 128, 256)
        l13_xyz, l13_points = self.sa_module_13(l12_xyz, l12_points)  # (B, 3, 256), (B, 128, 256)
        l22_xyz, l22_points, _ = self.sa_module_22(l1_xyz, l1_points)  # (B, 3, 256), (B, 256, 256)
        l23_xyz, l23_points = self.sa_module_23(l22_xyz, l22_points)  # (B, 3, 1), (B, out_dim, 1)
        l32_xyz, l32_points, _ = self.sa_module_32(l22_xyz, l22_points)  # (B, 3, 256), (B, 256, 256)
        l33_xyz, l33_points = self.sa_module_33(l32_xyz, l32_points)  # (B, 3, 1), (B, out_dim, 1)

        l13_points = l13_points.squeeze(2)
        l23_points = l23_points.squeeze(2)
        l33_points = l33_points.squeeze(2)
        l13_points = self.fc1(l13_points)
        l23_points = self.fc2(l23_points)
        l33_points = self.fc3(l33_points)
        logit1 = self.cls1(l13_points)
        logit2 = self.cls2(l23_points)
        logit3 = self.cls3(l33_points)
        return [logit1, logit2, logit3], [l13_points, l23_points, l33_points]


class PointNet_SD_Ensemble(nn.Module):
    def __init__(self, class_num):
        print('Downstream = PointNet-SD-Ensemble')
        super(PointNet_SD_Ensemble, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_21 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_22 = PointNet_SA_Module_KNN(128, 32, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_23 = PointNet_SA_Module_KNN(128, 48, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_31 = PointNet_SA_Module_KNN(None, None, 256, [512, 512], group_all=True, if_bn=False)
        self.sa_module_32 = PointNet_SA_Module_KNN(None, None, 256, [512, 512], group_all=True, if_bn=False)
        self.sa_module_33 = PointNet_SA_Module_KNN(None, None, 256, [512, 512], group_all=True, if_bn=False)

        self.fc1 = FCLayer(512, class_num)
        self.fc2 = FCLayer(512, class_num)
        self.fc3 = FCLayer(512, class_num)
        self.fc4 = FCLayer(1536, class_num)

    def forward(self, point_cloud):
        l0_xyz = point_cloud
        l0_points = point_cloud
        # print(f'PointNet point_cloud {point_cloud.shape}')

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        # print(f'PointNet 1 {l1_points.shape}')
        l21_xyz, l21_points, idx21 = self.sa_module_21(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l22_xyz, l22_points, idx22 = self.sa_module_22(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l23_xyz, l23_points, idx23 = self.sa_module_23(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        # print(f'PointNet 2 {l2_points.shape}')
        l31_xyz, l31_points = self.sa_module_31(l21_xyz, l21_points)  # (B, 3, 1), (B, out_dim, 1)
        l32_xyz, l32_points = self.sa_module_32(l22_xyz, l22_points)  # (B, 3, 1), (B, out_dim, 1)
        l33_xyz, l33_points = self.sa_module_33(l23_xyz, l23_points)  # (B, 3, 1), (B, out_dim, 1)
        # print(f'PointNet 3 {l33_points.shape}')
        l31_points = l31_points.squeeze(2)
        l32_points = l32_points.squeeze(2)
        l33_points = l33_points.squeeze(2)
        logit1 = self.fc1(l31_points)
        logit2 = self.fc2(l32_points)
        logit3 = self.fc3(l33_points)
        logit4 = self.fc4(torch.cat((l31_points, l32_points, l33_points), 1))
        # feat = [logit1, logit2, logit3]
        # logit1 = self.h1(logit1)
        # logit2 = self.h2(logit2)
        # logit3 = self.h3(logit3)
        return [logit1, logit2, logit3, logit4], [l31_points, l32_points, l33_points]
        # return [logit1, logit2, logit3], feat#[l31_points.squeeze(2), l32_points.squeeze(2), l33_points.squeeze(2)]#[l21_points, l22_points, l23_points]


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        # print(f'S, self.kneighbors: {S, self.kneighbors}')
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]
        points = points.contiguous()  # xyz [btach, points, xyz]
        """origin"""
        fps_idx = furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        """z2"""
        # new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(xyz, points, self.groups, False)
        # idx = idx.long().squeeze()
        # print(f'points, idx: {points.shape,idx.shape}')
        # grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        # new_xyz, new_points, grouped_xyz = new_xyz.contiguous(), new_points.contiguous(), grouped_xyz.contiguous()
        """z"""
        # new_xyz, fps_idx = fps_subsample_idx(xyz, self.groups)
        # new_points = gather_operation(points.permute(0, 2, 1).contiguous(), fps_idx)  # [B, d, npoint]
        # new_points = new_points.permute(0, 2, 1).contiguous()
        # print(f'new_points: {new_points.shape}')
        # idx = query_knn(self.kneighbors, xyz, new_xyz)

        # grouped_xyz = grouping_operation(xyz.permute(0, 2, 1).contiguous(), idx)  # [B, npoint, k, 3]
        # grouped_xyz = grouped_xyz.permute(0, 2, 3, 1).contiguous()
        # grouped_points = grouping_operation(points.permute(0, 2, 1).contiguous(), idx)  # [B, npoint, k, d]
        # grouped_points = grouped_points.permute(0, 2, 3, 1).contiguous()
        # print(f'grouped_xyz, grouped_points: {grouped_xyz.shape, grouped_points.shape}')
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        # print(f'new_points, grouped_points: {new_points.shape, grouped_points.shape}')
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = nn.Sequential(nn.ReLU(inplace=True),)
        # self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            nn.ReLU(inplace=True),
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        # z = self.net2(self.net1(x)) + x
        # print(f'z: {z.shape}')
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointMLP(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        print('Downstream = PointMLP')
        super(PointMLP, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=embed_dim, kernel_size=1, bias=True),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        # self.act = nn.ReLU(inplace=True),
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num)
        )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x, []


class PointMLP_SD(nn.Module): # classification-only model
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        print('Downstream = PointMLP-SD')
        super(PointMLP_SD, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=embed_dim, kernel_size=1, bias=True),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        fc_in_channels = []
        # self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            fc_in_channels.append(last_channel)
            # print(f'last_channel2: {last_channel}')

        # self.act = nn.ReLU(inplace=True),
        # self.act = get_activation(activation)
        self.fcs = nn.ModuleList()
        for i in range(len(pre_blocks)):
            self.fcs.append(nn.Sequential(
                    nn.Linear(fc_in_channels[i], 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, self.class_num)
                ))

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        inter_x = []
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            inter_x.append(x)

        logits = []
        for i, x in enumerate(inter_x):
            x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
            logit = self.fcs[i](x)
            logits.append(logit)
        return logits, inter_x


curve_config = {
    'short': [[100, 5], None, None],
    'default': [[100, 5], [100, 5], None, None],
    'long': [[10, 30], None, None, None]
}


class CurveNet(nn.Module):
    def __init__(self, num_classes=40, k=8, setting='default'):
        print('Downstream = CurveNet')
        super(CurveNet, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder original
        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])

        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][3])

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.conv2 = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        # self.fc = FCLayer(512, num_classes)

    def forward(self, xyz, return_feats=False):
        l0_points = self.lpfa(xyz, xyz)
        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        # simplified
        # l3_xyz, l3_points = self.cic31(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        # origin
        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)

        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        if return_feats:
            return x, None

        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)       
        x = self.dp1(x)
        x = self.conv2(x)

        # x = self.fc(x)
        return x, []


class CurveNet_SD(nn.Module): # classification-only model
    def __init__(self, num_classes=40, k=8, setting='default'):
        print('Downstream = CurveNet-SD')
        super(CurveNet_SD, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])

        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][3])

        self.conv01 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.conv02 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.conv03 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 2, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512 * 2, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024 * 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz, return_feats=False):
        l0_points = self.lpfa(xyz, xyz)
        l0_points = self.lpfa(xyz, xyz)
        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)
        x1 = self.conv01(l2_points)
        x_max1 = F.adaptive_max_pool1d(x1, 1)
        x_avg1 = F.adaptive_avg_pool1d(x1, 1)        
        x1 = torch.cat((x_max1, x_avg1), dim=1).squeeze(-1)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
        x2 = self.conv02(l3_points)
        x_max2 = F.adaptive_max_pool1d(x2, 1)
        x_avg2 = F.adaptive_avg_pool1d(x2, 1)
        x2 = torch.cat((x_max2, x_avg2), dim=1).squeeze(-1)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)
        x3 = self.conv03(l4_points)
        x_max3 = F.adaptive_max_pool1d(x3, 1)
        x_avg3 = F.adaptive_avg_pool1d(x3, 1)
        x3 = torch.cat((x_max3, x_avg3), dim=1).squeeze(-1)

        if return_feats:
            return x3, None

        logit1 = self.fc1(x1)
        logit2 = self.fc2(x2)
        logit3 = self.fc3(x3)

        return [logit1, logit2, logit3], [l2_points, l3_points, l4_points]


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1, dim_cat=2):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.dim_cat = dim_cat
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])

        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1)
        Q = self.mlp_2(feat_1)

        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)

        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))
        # print(f'SPD K_curr: {K_curr.shape}')
        # pred_label = self.shallow_cls1(K_curr) # shallow classifiers


        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)
        # print(f'SPD pcd_child: {pcd_child.shape}')
        # print(f'SPD delta: {delta.shape}')
        pcd_child = pcd_child + delta

        return pcd_child, K_curr#, pred_label


class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None, dim_cat=8):
        super(Decoder, self).__init__()
        self.num_p0 = num_p0
        self.dim_cat = dim_cat
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))

        self.uppers = nn.ModuleList(uppers)

        # self.deep_cls = PointNet(dim_cat) # pointnet2
        self.deep_cls = CurveNet(dim_cat, k=8) # curvenet
        # self.deep_cls = PointMLP(points=512, class_num=dim_cat, embed_dim=64, groups=1, res_expansion=1.0,
        #            activation="relu", bias=False, use_xyz=False, normalize="anchor",
        #            dim_expansion=[2, 2, 2], pre_blocks=[2, 2, 2], pos_blocks=[2, 2, 2],
        #            k_neighbors=[8, 16, 24], reducers=[2, 2, 2])   # pointmlp

    def forward(self, feat, partial, return_P0=False):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd, pred_labels, cls_feats = [], [], []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # seed: (B, num_pc, 3)
        # print(f'Decoder seed: {pcd.shape}')
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0) # seed_fps: (B, num_p0, 3)
        # print(f'Decoder seed after fps: {pcd.shape}')

        if return_P0:
            arr_pcd.append(pcd)
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous() #B, 3, 512
        # print(f'Decoder pcd: {pcd.shape}')

        for i, upper in enumerate(self.uppers):
            # print(f'upper iter: {i}')
            pcd, K_prev = upper(pcd, feat, K_prev)
            # pred_label, cls_feat = self.classifiers[i](pcd)
            # print(f'upper pred_label: {pred_label.shape}')
            # pred_labels.append(pred_label)
            # cls_feats.append(cls_feat)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        #need one more layer for deep classifier
        # print(f'Decoder pcd: {pcd.shape}')
        pred_labels, cls_feats = self.deep_cls(pcd)
        # pred_labels.append(pred_label)
        # cls_feats.append(cls_feat)

        return arr_pcd, pred_labels, cls_feats


class SnowflakeNet(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, ncat=40, up_factors=None):
        """
        Args:
            dim_feat: int, dimension of global feature
            num_pc: int
            num_p0: int
            radius: searching radius
            up_factors: list of int
        """
        super(SnowflakeNet, self).__init__()
        self.feat_extractor = FeatureExtractor(out_dim=dim_feat)
        self.decoder = Decoder(dim_feat=dim_feat, num_pc=num_pc, num_p0=num_p0, radius=radius, up_factors=up_factors, dim_cat=ncat)

    def forward(self, point_cloud, return_P0=False):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        pcd_bnc = point_cloud
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        code = self.feat_extractor(point_cloud)
        # print(f'SnowflakeNet code: {code.shape}')
        # print(f'SnowflakeNet pcd_bnc: {pcd_bnc.shape}')
        out, pred_labels, cls_feats = self.decoder(code, pcd_bnc, return_P0=return_P0)
        # print(f'SnowflakeNet cls_feats: {cls_feats[2].shape}')
        return out, pred_labels, [code.squeeze(2)]+cls_feats
