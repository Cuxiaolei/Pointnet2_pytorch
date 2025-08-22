import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, feature_interpolation


# 空间变换网络（STN）用于生成3x3变换矩阵
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)  # 3x3矩阵的扁平表示
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: [B, 3, N] 仅使用坐标信息
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 初始化为单位矩阵
        iden = torch.eye(3, device=x.device).view(1, 9).repeat(x.size(0), 1)
        x = x + iden
        x = x.view(-1, 3, 3)  # 转换为3x3矩阵 [B, 3, 3]
        return x


class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        # 添加空间变换网络
        self.stn = STN3d()
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=9, mlp=[32, 32, 64],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=0.8, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512],
                                          group_all=False)

        # 特征传播层
        self.fp4 = nn.Sequential(
            nn.Conv1d(512 + 256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fp3 = nn.Sequential(
            nn.Conv1d(256 + 128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fp2 = nn.Sequential(
            nn.Conv1d(128 + 64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fp1 = nn.Sequential(
            nn.Conv1d(70, 64, 1),  # 6 + 64 = 70 通道
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, num_class, 1)
        )

    def forward(self, xyz):
        # 提取坐标用于空间变换
        l0_coords = xyz[:, :3, :]  # [B, 3, N]
        # 应用空间变换网络
        trans = self.stn(l0_coords)  # [B, 3, 3]
        # 对坐标进行变换
        l0_coords_transformed = torch.bmm(trans, l0_coords)  # [B, 3, N]

        # 重组输入特征（变换后的坐标 + 原始特征）
        l0_xyz = l0_coords_transformed
        l0_points = xyz[:, 3:, :]  # [B, 6, N] - 颜色+法向量

        # 下采样阶段
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B, 3, 1024], [B, 64, 1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 3, 256], [B, 128, 256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 3, 64], [B, 256, 64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [B, 3, 16], [B, 512, 16]

        # 特征传播阶段
        l4_points_interpolated = feature_interpolation(l3_xyz, l4_xyz, l4_points)  # [B, 512, 64]
        l3_points = self.fp4(torch.cat([l3_points, l4_points_interpolated], dim=1))  # [B, 256, 64]

        l3_points_interpolated = feature_interpolation(l2_xyz, l3_xyz, l3_points)  # [B, 256, 256]
        l2_points = self.fp3(torch.cat([l2_points, l3_points_interpolated], dim=1))  # [B, 128, 256]

        l2_points_interpolated = feature_interpolation(l1_xyz, l2_xyz, l2_points)  # [B, 128, 1024]
        l1_points = self.fp2(torch.cat([l1_points, l2_points_interpolated], dim=1))  # [B, 64, 1024]

        l1_points_interpolated = feature_interpolation(l0_xyz, l1_xyz, l1_points)  # [B, 64, N]
        l0_points = self.fp1(torch.cat([l0_points, l1_points_interpolated], dim=1))  # [B, num_class, N]

        # 返回预测结果和变换矩阵（作为trans_feat）
        return l0_points, trans


class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.weight = weight  # 类别权重

    def forward(self, pred, target, trans_feat):
        # 交叉熵损失（pred: [B, C, N], target: [B, N]）
        ce_loss = F.cross_entropy(pred, target, weight=self.weight)

        # 变换一致性损失（trans_feat是3x3矩阵）
        trans_loss = torch.norm(
            torch.eye(3, device=trans_feat.device) - torch.bmm(trans_feat, trans_feat.transpose(2, 1)))

        # 总损失 = 交叉熵损失 + 变换损失
        return ce_loss + 0.001 * trans_loss


def get_loss_function(weight=None):
    return get_loss(weight)
