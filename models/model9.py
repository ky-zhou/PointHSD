import torch
import torch.nn as nn
from models.utils import *
from models.skip_transformer import SkipTransformer
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from models.pointmlp import index_points, knn_point


class FCLayer(nn.Module):
    def __init__(self, in_channel) :
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel//2),
            nn.BatchNorm1d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_channel//2, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(in_channel//4, class_num)
        )

    def forward(self, code):
        return self.fc(code)


class ConvLayer(nn.Module):
    def __init__(self, in_channel) :
        super(ConvLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, 1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(in_channel//2, in_channel//4),
            # nn.BatchNorm1d(in_channel//4),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
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


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
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
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
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

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = nn.Sequential(nn.ReLU(inplace=True),)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = nn.Sequential(nn.ReLU(inplace=True),)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
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
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
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


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points


class PointMLP_SD_Seg(nn.Module):
    def __init__(self, num_classes=50, points=1024, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2], pre_blocks=[2, 2, 2], pos_blocks=[2, 2, 2],
                 k_neighbors=[16, 24, 32], reducers=[2, 2, 2],
                 de_dims=[512, 256, 128], de_blocks=[4, 4, 4], gmp_dim=64, cls_dim=64):
        super(PointMLP_SD_Seg, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = num_classes
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]
        ### Building Encoder #####
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
            en_dims.append(last_channel)

        print(f'MLP en_dims: {en_dims}')
        ### Building Decoder #####
        self.decode_list1 = nn.ModuleList()
        self.decode_list2 = nn.ModuleList()
        self.decode_list3 = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0, en_dims[0])
        print(f'MLP de_dims: {de_dims}')
        assert len(en_dims) ==len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-3):
            self.decode_list1.append( # 512+64->128
                PointNetFeaturePropagation(de_dims[i*3]+en_dims[i+3], de_dims[i+3],
                                           blocks=de_blocks[i+2], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation))
        for i in range(len(en_dims)-2):
            self.decode_list2.append( # 512+128->256; 256+64->128
                PointNetFeaturePropagation(de_dims[i*2]+en_dims[i+2], de_dims[i+2],
                                           blocks=de_blocks[i+1], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation))
        for i in range(len(en_dims)-1):
            self.decode_list3.append( # 512+256->512; 512+128->256; 256+64->128
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                           blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation))
        self.act = nn.ReLU(inplace=True),
        # class label mapping
        self.cls_map = nn.Sequential(
            ConvBNReLU1D(16, cls_dim, bias=bias, activation=activation),
            # ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation)
        )
        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = nn.ModuleList()
        self.gmp_map_end.append(ConvBNReLU1D(gmp_dim*(len(en_dims)-2), gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end.append(ConvBNReLU1D(gmp_dim*(len(en_dims)-1), gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end.append(ConvBNReLU1D(gmp_dim*len(en_dims), gmp_dim, bias=bias, activation=activation))

        # classifier
        self.classifier = nn.ModuleList()
        for i in range(len(k_neighbors)):
            self.classifier.append(nn.Sequential(
                nn.Conv1d(gmp_dim+cls_dim+de_dims[-1], 128, 1, bias=bias),
                nn.BatchNorm1d(128),
                nn.Dropout(),
                nn.Conv1d(128, num_classes, 1, bias=bias)))
        self.en_dims = en_dims

    def forward(self, pcd, cls_label):
        xyz = pcd[-1]
        x = self.embedding(pcd[-1].permute(0, 2, 1))  # B,D,N

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, D, N]
        inter_x = [] # with varying ks

        # here is the encoder
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            xyz_list.append(xyz)
            x_list.append(x)
            inter_x.append(x)
            # print(f'MLP encoder x: {i, x.shape}')#b, 128, 512; b, 256, 256; b, 512, 128

        # here is the decoder
        xyz_list.reverse() # smaller to larger sizes: 128->256->512->N
        x_list.reverse() # smaller to larger sizes: 128->256->512->N
        x = x_list[0]
        decoded_xs = []

        for i in range(len(self.decode_list1)):
            # 256, 3; 128, 3; 256, 256; 512, 128: first two: N,3, last two: D,N
            # print(f'Into decoder1 {i, xyz_list[i+3].shape, xyz_list[i*3].shape, x_list[i+3].shape, x.shape}')
            x = self.decode_list1[i](xyz_list[i+3], xyz_list[i*3], x_list[i+3], x)
            # print(f'decoder1 x2 {i, x.shape}') # D,N: 512, 256; 256, 512; 128, 1024
        decoded_xs.append(x) #1
        x = x_list[0]

        for i in range(len(self.decode_list2)):
            # 256, 3; 128, 3; 256, 256; 512, 128: first two: N,3, last two: D,N
            # print(f'Into decoder2 {i, xyz_list[i+2].shape, xyz_list[i*2].shape, x_list[i+2].shape, x.shape}')
            x = self.decode_list2[i](xyz_list[i+2], xyz_list[i*2], x_list[i+2], x)
            # print(f'decoder2 x2 {i, x.shape}') # D,N: 512, 256; 256, 512; 128, 1024
        decoded_xs.append(x) #2
        x = x_list[0]

        for i in range(len(self.decode_list3)):
            # 256, 3; 128, 3; 256, 256; 512, 128: first two: N,3, last two: D,N
            # print(f'Into decoder3 {i, xyz_list[i+1].shape, xyz_list[i].shape, x_list[i+1].shape, x.shape}')
            x = self.decode_list3[i](xyz_list[i+1], xyz_list[i], x_list[i+1], x)
            # print(f'decoder3 x2 {i, x.shape}') # D,N: 512, 256; 256, 512; 128, 1024
        decoded_xs.append(x) #3

        # here is the global context
        gmp_list = {0: [], 1: [], 2: []}
        for i in range(len(x_list)):
            x_max = F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1)
            # print(f'x_max: {i, x_max.shape}') #[4, 64, 1]
            if i < 2:
                gmp_list[0].append(x_max)
            if i < 3:
                gmp_list[1].append(x_max)
            gmp_list[2].append(x_max)
        global_context = []
        for i in range(len(decoded_xs)):
            global_context.append(self.gmp_map_end[i](torch.cat(gmp_list[i], dim=1))) # [b, gmp_dim, 1]
            # print(f'global_context: {global_context[-1].shape}') #[4, 64, 1]

        #here is the cls_token
        cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))  # [b, cls_dim, 1]
        logits = []
        for i, dx in enumerate(decoded_xs):
            x = torch.cat([dx, global_context[i].repeat([1, 1, dx.shape[-1]]), cls_token.repeat([1, 1, dx.shape[-1]])], dim=1)
            # print(f'Into FC: {x.shape}') # 256, 1024
            x = self.classifier[i](x)
            logits.append(x)
        return logits, global_context


class PointNet_SD_Seg(nn.Module):
    def __init__(self, class_num):
        print('Downstream = PointNet-SD-Seg')
        super(PointNet_SD_Seg, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(1024, 12, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_12 = PointNet_SA_Module_KNN(512, 8, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_13 = PointNet_SA_Module_KNN(None, None, 256, [256, 512], group_all=True, if_bn=False)

        self.sa_module_22 = PointNet_SA_Module_KNN(512, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_23 = PointNet_SA_Module_KNN(None, None, 256, [512, 512], group_all=True, if_bn=False)

        self.sa_module_32 = PointNet_SA_Module_KNN(512, 24, 256, [256, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_33 = PointNet_SA_Module_KNN(None, None, 256, [512, 1024], group_all=True, if_bn=False)

        self.fp31 = PointNet_FP_Module(256, [256, 128], True, 22)
        self.fp32 = PointNet_FP_Module(256, [256, 256], True, 128)
        self.fp33 = PointNet_FP_Module(1024, [512, 256], True, 256)

        self.fp21 = PointNet_FP_Module(256, [256, 128], True, 22)
        self.fp22 = PointNet_FP_Module(256, [256, 256], True, 128)
        self.fp23 = PointNet_FP_Module(512, [256, 256], True, 256)

        self.fp11 = PointNet_FP_Module(256, [256, 128], True, 22)
        self.fp12 = PointNet_FP_Module(256, [256, 256], True, 128)
        self.fp13 = PointNet_FP_Module(512, [256, 256], True, 256)

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, class_num, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, class_num, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, class_num, 1)
        )
        self.fc1 = FCLayer(512)
        self.fc2 = FCLayer(512)
        self.fc3 = FCLayer(1024)

    def forward(self, point_cloud, cls_label):
        l0_xyz = point_cloud[-1].permute(0, 2, 1).contiguous()
        l0_points = point_cloud[-1].permute(0, 2, 1).contiguous()
        B,N,C = point_cloud[-1].size()
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        # print(f'PointNetSeg l0_points {l0_points.shape}')
        
        # print(f'PointNet point_cloud {point_cloud.shape}')
        l1_xyz, l1_points, _ = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        # print(f'PointNetSeg l1_xyz {l1_xyz.shape, l1_points.shape}')
        l12_xyz, l12_points, _ = self.sa_module_12(l1_xyz, l1_points)  # (B, 3, 256), (B, 128, 256)
        l13_xyz, l13_points = self.sa_module_13(l12_xyz, l12_points)  # (B, 3, 1), (B, 512, 1)

        l0_points = torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1)
        l12_points = self.fp13(l12_xyz, l13_xyz, l12_points, l13_points) # ->512*256
        # print(f'PointNetSeg l12_points {l12_points.shape}')
        l11_points = self.fp12(l1_xyz, l12_xyz, l1_points, l12_points) # ->2048*128
        # print(f'PointNetSeg l11_points {l11_points.shape}')
        l10_points = self.fp11(l0_xyz, l1_xyz, l0_points, l11_points) # ->2048*128
        # print(f'PointNetSeg l10_points {l10_points.shape}')

        l22_xyz, l22_points, _ = self.sa_module_22(l1_xyz, l1_points)  # (B, 3, 256), (B, 256, 256)
        l23_xyz, l23_points = self.sa_module_23(l22_xyz, l22_points)  # (B, 3, 1), (B, out_dim, 1)
        
        l22_points = self.fp23(l22_xyz, l23_xyz, l22_points, l23_points) # ->512*256
        # print(f'PointNetSeg l22_points {l22_points.shape}')
        l21_points = self.fp22(l1_xyz, l22_xyz, l1_points, l22_points) # ->1024*256
        # print(f'PointNetSeg l21_points {l21_points.shape}')
        l20_points = self.fp21(l0_xyz, l1_xyz, l0_points, l21_points) # ->2048*128
        # print(f'PointNetSeg l20_points {l20_points.shape}')

        l32_xyz, l32_points, _ = self.sa_module_32(l22_xyz, l22_points)  # (B, 3, 256), (B, 256, 256)
        l33_xyz, l33_points = self.sa_module_33(l32_xyz, l32_points)  # (B, 3, 1), (B, out_dim, 1)
        # print(f'PointNetSeg 1 {l22_xyz.shape, l32_xyz.shape, l22_points.shape, l32_points.shape}')

        l32_points = self.fp33(l32_xyz, l33_xyz, l32_points, l33_points) # ->512*256
        # print(f'PointNetSeg l32_points {l32_points.shape}')
        l31_points = self.fp32(l1_xyz, l32_xyz, l1_points, l32_points) # ->1024*256
        # print(f'PointNetSeg l31_points {l31_points.shape}')
        l30_points = self.fp31(l0_xyz, l1_xyz, l0_points, l31_points) # ->2048*128
        # print(f'PointNetSeg l30_points {l30_points.shape}')

        l13_points = l13_points.squeeze(2)
        l23_points = l23_points.squeeze(2)
        l33_points = l33_points.squeeze(2)

        info1 = self.fc1(l13_points)
        info2 = self.fc2(l23_points)
        info3 = self.fc3(l33_points)

        logit1 = self.conv1(l10_points)
        logit2 = self.conv2(l20_points)
        logit3 = self.conv3(l30_points)
        # print(f'PointNetSeg logits {logit1.shape, logit2.shape, logit3.shape}')
        return [logit1, logit2, logit3], [info1, info2, info3]


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

        # self.deep_cls = PointNet_SD_Seg(dim_cat)
        self.deep_cls = PointMLP_SD_Seg(num_classes=dim_cat, points=1024, embed_dim=64, groups=1, res_expansion=1.0,
                 bias=False, use_xyz=False, normalize="anchor")

    def forward(self, feat, partial, cls_label=None, return_P0=False):
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
        pred_labels, cls_feats = self.deep_cls(arr_pcd, cls_label)
        # pred_labels.append(pred_label)
        # cls_feats.append(cls_feat)

        return arr_pcd, pred_labels, cls_feats


class SnowflakeNet(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, dim_cat=50, up_factors=None):
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
        self.decoder = Decoder(dim_feat=dim_feat, num_pc=num_pc, num_p0=num_p0, radius=radius, up_factors=up_factors, dim_cat=dim_cat)

    def forward(self, point_cloud, cls_label=None, return_P0=False):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        pcd_bnc = point_cloud
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        code = self.feat_extractor(point_cloud)
        # print(f'SnowflakeNet code: {code.shape}')
        # print(f'SnowflakeNet pcd_bnc: {pcd_bnc.shape}')
        out, pred_labels, cls_feats = self.decoder(code, pcd_bnc, cls_label, return_P0=return_P0)
        return out, pred_labels, cls_feats
