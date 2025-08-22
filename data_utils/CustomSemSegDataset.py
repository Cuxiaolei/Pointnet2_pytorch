import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


def pc_normalize(pc):
    """点云归一化处理"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def random_point_dropout(pc, label, max_dropout_ratio=0.875):
    """随机丢弃点云点（数据增强）"""
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx] = pc[0]  # 用第一个点填充
        label[drop_idx] = label[0]
    return pc, label


def rotate_point_cloud_z(pc):
    """绕Z轴旋转（数据增强）"""
    theta = np.random.uniform(0, np.pi * 2)
    cosz = np.cos(theta)
    sinz = np.sin(theta)
    rot_mat = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
    pc[:, :3] = np.dot(pc[:, :3], rot_mat)
    return pc


class CustomSemSegDataset(Dataset):
    def __init__(self, root, split, num_point, block_size=1.0, sample_rate=1.0, transform=True):
        self.root = root
        self.split = split
        self.num_point = num_point
        self.block_size = block_size  # 分块大小
        self.sample_rate = sample_rate
        self.transform = transform  # 是否使用数据增强
        self.label_weights = None

        # 读取划分文件
        split_file = os.path.join(root, f'{split}_scenes.txt')
        with open(split_file, 'r') as f:
            self.scene_list = [line.strip() for line in f.readlines()]

        # 加载所有场景数据
        self.data_list = []
        self.label_list = []
        for scene_path in self.scene_list:
            full_path = os.path.join(root, scene_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"场景文件不存在: {full_path}")

            # 加载npy文件 (每个点: [x,y,z,r,g,b,nx,ny,nz,label])
            scene_data = np.load(full_path)
            points = scene_data[:, :9]  # 前9通道: 坐标+颜色+法向量
            labels = scene_data[:, 9].astype(np.int32)  # 第10通道是标签

            self.data_list.append(points)
            self.label_list.append(labels)

        # 计算标签权重（用于不平衡数据）
        all_labels = np.concatenate(self.label_list, axis=0)
        self.label_weights = np.bincount(all_labels)
        self.label_weights = self.label_weights / np.sum(self.label_weights)
        self.label_weights = 1.0 / (np.log(1.2 + self.label_weights))

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        points = self.data_list[idx].copy()
        labels = self.label_list[idx].copy()

        # 点云归一化（仅对坐标）
        points[:, :3] = pc_normalize(points[:, :3])

        # 数据增强（仅训练集）
        if self.split == 'train' and self.transform:
            # 随机旋转
            points = rotate_point_cloud_z(points)
            # 随机丢弃点
            points, labels = random_point_dropout(points, labels)

        # 如果点数量超过需要的数量，随机采样
        if points.shape[0] > self.num_point:
            choice = np.random.choice(points.shape[0], self.num_point, replace=False)
            points = points[choice, :]
            labels = labels[choice]
        # 如果点数量不足，重复填充
        elif points.shape[0] < self.num_point:
            repeat = self.num_point // points.shape[0] + 1
            points = np.tile(points, (repeat, 1))[:self.num_point, :]
            labels = np.tile(labels, (repeat,))[:self.num_point]

        # 转换为Tensor
        points = torch.FloatTensor(points.T)  # 模型期望的形状是 [C, N]
        labels = torch.LongTensor(labels)

        return points, labels