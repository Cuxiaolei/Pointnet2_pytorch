import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=9, mlp=[32, 32, 64],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=0.8, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512],
                                          group_all=False)
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
            nn.Conv1d(64 + 9, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, num_class, 1)
        )

    def forward(self, xyz):
        # Set Abstraction layers
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz[:, 3:, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature Propagation layers
        l3_points = self.fp4(torch.cat([l3_points, l4_points], dim=1))
        l2_points = self.fp3(torch.cat([l2_points, l3_points], dim=1))
        l1_points = self.fp2(torch.cat([l1_points, l2_points], dim=1))
        l0_points = self.fp1(torch.cat([l0_points, l1_points], dim=1))

        return l0_points, l4_points


# 修复损失函数定义，确保正确接收权重参数
class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.weight = weight  # 类别权重

    def forward(self, pred, target, trans_feat):
        # 交叉熵损失
        ce_loss = F.cross_entropy(pred, target, weight=self.weight)

        # 变换一致性损失
        trans_loss = torch.norm(
            torch.eye(3, device=trans_feat.device) - torch.bmm(trans_feat, trans_feat.transpose(2, 1)))

        # 总损失 = 交叉熵损失 + 变换损失
        return ce_loss + 0.001 * trans_loss


# 修改损失函数获取方式，使其能接收权重参数
def get_loss_function(weight=None):
    return get_loss(weight)


if __name__ == '__main__':
    import  torch
    model = get_model(3)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))