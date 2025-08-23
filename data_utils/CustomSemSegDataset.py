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
    if m > 0:  # 防止除以零
        pc = pc / m
    return pc


def random_point_dropout(pc, label, max_dropout_ratio=0.3):  # 从0.875→0.3
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx] = pc[0]
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


def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    """随机缩放（新增数据增强）"""
    scale = np.random.uniform(scale_low, scale_high)
    pc[:, :3] *= scale
    return pc


def random_shift_point_cloud(pc, shift_range=0.1):
    """随机平移（新增数据增强）"""
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    pc[:, :3] += shifts
    return pc


class CustomSemSegDataset(Dataset):
    def __init__(self, root, split, num_point, block_size=1.0, sample_rate=1.0, transform=True):
        self.root = root
        self.split = split
        self.num_point = num_point  # 每个样本的点数量
        self.block_size = block_size  # 分块大小
        self.sample_rate = sample_rate  # 采样率
        self.transform = transform  # 是否使用数据增强
        self.label_weights = None

        # 读取划分文件
        split_file = os.path.join(root, f'{split}_scenes.txt')
        with open(split_file, 'r') as f:
            self.scene_list = [line.strip() for line in f.readlines()]

        # 预加载场景信息（不加载完整数据，只记录文件路径和点数量）
        self.scene_paths = []
        self.scene_point_counts = []

        for scene_path in self.scene_list:
            full_path = os.path.join(root, scene_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"场景文件不存在: {full_path}")

            # 修复：不使用with语句加载mmap文件
            data = np.load(full_path, mmap_mode='r')
            self.scene_point_counts.append(data.shape[0])
            self.scene_paths.append(full_path)
            # 关闭文件句柄
            del data

        # 计算每个场景的采样权重（基于点数量）
        self.scene_weights = np.array(self.scene_point_counts) / np.sum(self.scene_point_counts)

        # 预计算标签权重
        self._compute_label_weights()

    def _compute_label_weights(self):
        cache_file = os.path.join(self.root, f'{self.split}_label_weights.npy')
        if os.path.exists(cache_file):
            os.remove(cache_file)  # 删除旧缓存，确保新权重生效

        # 1. 统计所有类别的总点数（采样10%点，兼顾效率）
        all_labels = []
        for path in self.scene_paths:
            data = np.load(path, mmap_mode='r')
            sample_size = max(1000, int(data.shape[0] * 0.1))
            indices = np.random.choice(data.shape[0], sample_size, replace=False)
            all_labels.append(data[indices, 9].astype(np.int32))
            del data
        all_labels = np.concatenate(all_labels)

        # 2. 计算激进权重：总点数 / 类别点数（类别越少，权重越大）
        cls_counts = np.bincount(all_labels, minlength=3)
        total_counts = cls_counts.sum()
        self.label_weights = total_counts / (cls_counts + 1e-6)  # 避免除零
        # 归一化权重（防止权重过大导致训练不稳定）
        self.label_weights = self.label_weights / self.label_weights.max() * 10  # 最大权重设为10

        np.save(cache_file, self.label_weights)
        print(f"类别权重：{self.label_weights}")  # 确保类别2权重最大（如[3.5, 1.8, 5.2]）
        return self.label_weights



    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        # 加载单个场景数据（修复mmap加载问题）
        scene_path = self.scene_paths[idx]
        # 不使用with语句，直接加载
        data = np.load(scene_path, mmap_mode='r')

        # 随机采样点（针对20W点的大场景）
        num_points = data.shape[0]

        # 如果点数量远大于需要的数量，先进行一次粗采样
        if num_points > self.num_point * 5:
            sample_size = int(self.num_point * 2)  # 先采样到目标的2倍
            indices = np.random.choice(num_points, sample_size, replace=False)
            points = data[indices, :9].copy()  # 前9通道: 坐标+颜色+法向量
            labels = data[indices, 9].astype(np.int32).copy()
        else:
            points = data[:, :9].copy()
            labels = data[:, 9].astype(np.int32).copy()

        # 关闭文件句柄
        del data

        # 点云归一化（仅对坐标）
        points[:, :3] = pc_normalize(points[:, :3])

        # 数据增强部分（train模式下）
        if self.split == 'train' and self.transform:
            points = rotate_point_cloud_z(points)  # 保留旋转（不改变类别）
            points = random_scale_point_cloud(points)  # 保留缩放（全局，不破坏局部）
            points = random_shift_point_cloud(points)  # 保留平移（全局，不破坏局部）
            points, labels = random_point_dropout(points, labels)  # 用降低后的丢弃比例

        # 最终采样到目标点数
        if points.shape[0] > self.num_point:
            # 对大场景使用分层采样，保证类别分布
            choice = self._stratified_sampling(labels, self.num_point)
            points = points[choice, :]
            labels = labels[choice]
        # 如果点数量不足，重复填充
        elif points.shape[0] < self.num_point:
            repeat = self.num_point // points.shape[0] + 1
            points = np.tile(points, (repeat, 1))[:self.num_point, :]
            labels = np.tile(labels, (repeat,))[:self.num_point]

            # 最终采样后，添加类别分布日志
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        # 只在调试时打印，避免日志过多（可每100个样本打印一次）
        if idx % 100 == 0:
            print(f"样本{idx}类别分布: {label_dist}")  # 后续可改为logger.info

        # 转换为Tensor (模型期望的形状是 [C, N])
        points = torch.FloatTensor(points.T)
        labels = torch.LongTensor(labels)

        return points, labels

    def _stratified_sampling(self, labels, num_samples):
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)

        # 1. 每个类别平均分配点数（如8192点→3类各2730/2731点）
        base_points = num_samples // num_classes
        remain_points = num_samples % num_classes
        samples_per_cls = {cls: base_points for cls in unique_labels}
        # 剩余点数随机分给前N个类，保证总数达标
        for cls in list(unique_labels)[:remain_points]:
            samples_per_cls[cls] += 1

        indices = []
        for cls in unique_labels:
            # 提取当前类的所有点索引
            cls_indices = np.where(labels == cls)[0]
            available = len(cls_indices)
            need = samples_per_cls[cls]

            if available >= need:
                # 点数充足：随机采need个
                selected = np.random.choice(cls_indices, need, replace=False)
            else:
                # 点数不足：先取全部，剩余重复采样（保证数量）
                selected = np.concatenate([
                    cls_indices,
                    np.random.choice(cls_indices, need - available, replace=True)
                ])
            indices.extend(selected)

        # 打乱顺序，避免同类点聚集
        indices = np.array(indices)
        np.random.shuffle(indices)
        return indices
