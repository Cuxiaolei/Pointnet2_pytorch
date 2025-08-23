"""
适配自定义3分类任务的测试代码
与train_semseg.py保持参数和逻辑一致
"""
import argparse
import os
import torch
import logging
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从训练代码同步导入模型和数据集
from models import pointnet2_sem_seg
from data_utils.CustomSemSegDataset import CustomSemSegDataset

# 3分类任务的类别信息（与训练代码保持一致）
classes = ['0', '1', '2']  # 替换为实际类别名称
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for cat, i in class2label.items()}

# 类别颜色映射（可根据需要调整）
def g_label2color(label):
    color_map = [
        [255, 0, 0],    # class0: 红色
        [0, 255, 0],    # class1: 绿色
        [0, 0, 255]     # class2: 蓝色
    ]
    return color_map[label]
# 在测试数据集加载后，添加投票时的随机变换函数
def random_rotate_for_voting(pc):
    """投票时的轻微随机旋转（仅绕Z轴）"""
    theta = np.random.uniform(-np.pi/12, np.pi/12)  # 小角度旋转
    cosz = np.cos(theta)
    sinz = np.sin(theta)
    rot_mat = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
    pc[:, :3] = np.dot(pc[:, :3], rot_mat)  # 仅旋转坐标
    return pc
# 训练时的归一化函数（来自CustomSemSegDataset）
def pc_normalize(pc):
    """训练时使用的坐标归一化：中心化并缩放到单位球"""
    centroid = np.mean(pc, axis=0)  # 计算点云中心
    pc = pc - centroid  # 中心化（所有点减去中心点坐标）
    m = np.max(np.sqrt(np.sum(pc **2, axis=1)))  # 计算点到中心的最大距离
    if m > 0:
        pc = pc / m  # 缩放到单位球（坐标范围[-1, 1]）
    return pc
def parse_args():
    parser = argparse.ArgumentParser('PointNet++ Semantic Segmentation Testing')
    # 与训练代码同步的默认参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备')
    parser.add_argument('--num_point', type=int, default=60000, help='每个样本的点数量')
    parser.add_argument('--log_dir', type=str, default='custom_sem_seg_logs', help='日志和模型根目录')
    parser.add_argument('--visual', action='store_true', default=False, help='是否可视化结果')
    parser.add_argument('--num_votes', type=int, default=3, help='投票聚合次数')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='不使用CUDA')
    # 数据集路径（与训练代码默认路径一致）
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/data/data_s3dis_pointNeXt', help='数据集根目录')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # 设备配置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # 日志和可视化目录（与训练代码目录结构一致）
    experiment_dir = args.log_dir  # 默认为custom_sem_seg_logs
    # 自动查找最新的训练日志目录（按时间戳排序）
    if os.path.isdir(experiment_dir):
        subdirs = [d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d))]
        if subdirs:
            subdirs.sort(reverse=True)  # 按时间戳倒序，取最新的
            experiment_dir = os.path.join(experiment_dir, subdirs[0])
    visual_dir = os.path.join(experiment_dir, 'visual')
    Path(visual_dir).mkdir(parents=True, exist_ok=True)

    # 日志设置
    logger = logging.getLogger("ModelTest")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'test_eval.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('参数配置:')
    log_string(args)

    # 与训练代码保持一致的类别数
    NUM_CLASSES = 3
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    # 加载测试数据集（使用自定义数据集）
    log_string("加载测试数据...")
    TEST_DATASET = CustomSemSegDataset(
        root=args.dataset_root,
        split='test',
        num_point=NUM_POINT,
        block_size=1.5,
        sample_rate=1.0,
        transform=False
    )
    test_loader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.cuda else False
    )
    log_string(f"测试集样本数: {len(TEST_DATASET)}")

    # 加载模型（与训练代码一致）
    log_string("加载模型...")
    model = pointnet2_sem_seg.get_model(NUM_CLASSES)
    if args.cuda:
        model = model.cuda()

    # 自动查找最佳模型（训练代码保存的是state_dict）
    best_model_path = None
    for file in os.listdir(experiment_dir):
        if file.startswith('best_model') and file.endswith('.pth'):
            best_model_path = os.path.join(experiment_dir, file)
            break
    if not best_model_path:
        log_string("未找到最佳模型文件！")
        return

    model.load_state_dict(torch.load(best_model_path, map_location='cuda' if args.cuda else 'cpu'))
    model.eval()
    log_string(f"已加载模型: {best_model_path}")

    with torch.no_grad():
        # 初始化评估指标
        total_seen_class = [0] * NUM_CLASSES
        total_correct_class = [0] * NUM_CLASSES
        total_iou_deno_class = [0] * NUM_CLASSES

        # 用于存储所有预测和标签
        all_preds = []
        all_targets = []
        all_points = []

        log_string('---- 开始测试评估 ----')

        # 批次处理测试数据
        for batch_idx, (points, target) in enumerate(tqdm(test_loader, desc="测试进度")):
            # 点云形状：[B, C, N]，其中C=9（坐标3+颜色3+法向量3）

            # 1. 提取坐标通道（前3维），转为numpy
            points_np = points.numpy()  # [B, C, N]
            coord = points_np[:, :3, :]  # [B, 3, N]，仅坐标通道

            # 2. 展平坐标后归一化（与训练时单样本处理逻辑一致）
            coord_flat = coord.reshape(-1, 3)  # [B*N, 3]
            coord_normalized = pc_normalize(coord_flat)  # 应用与训练相同的归一化
            points_np[:, :3, :] = coord_normalized.reshape(coord.shape)  # 还原形状

            # 3. 转回Tensor，确保数据类型和设备正确
            points = torch.FloatTensor(points_np)
            if args.cuda:
                points = points.cuda()
                target = target.cuda()

            # 4. 模型推理（输入形状[B, C, N]，与训练一致）
            pred_sum = None
            for _ in range(args.num_votes):
                seg_pred, _ = model(points)  # 无需转置，直接使用[B, C, N]
                if pred_sum is None:
                    pred_sum = seg_pred
                else:
                    pred_sum += seg_pred

            # 平均投票结果
            seg_pred = pred_sum / args.num_votes
            pred_choice = seg_pred.contiguous().cpu().data.max(1)[1].numpy()  # [B, N]
            target_np = target.cpu().numpy()  # [B, N]
            points_np = points[:, :3, :].cpu().numpy().transpose(0, 2, 1)  # [B, N, 3]  # [B, N, 3]

            # 收集结果
            all_preds.append(pred_choice)
            all_targets.append(target_np)
            all_points.append(points_np)

            # 计算每类指标
            for l in range(NUM_CLASSES):
                total_seen = np.sum(target_np == l)
                total_correct = np.sum((pred_choice == l) & (target_np == l))
                total_iou_deno = np.sum((pred_choice == l) | (target_np == l))

                total_seen_class[l] += total_seen
                total_correct_class[l] += total_correct
                total_iou_deno_class[l] += total_iou_deno

        # 合并所有结果
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_points = np.concatenate(all_points, axis=0)

        # 计算整体评估指标
        IoU = np.array([
            total_correct_class[l] / (total_iou_deno_class[l] + 1e-6)
            for l in range(NUM_CLASSES)
        ])
        acc_per_class = np.array([
            total_correct_class[l] / (total_seen_class[l] + 1e-6)
            for l in range(NUM_CLASSES)
        ])
        overall_acc = np.sum(total_correct_class) / (np.sum(total_seen_class) + 1e-6)
        mean_iou = np.mean(IoU[IoU > 0])  # 忽略无样本的类别

        # 输出评估结果
        log_string("\n------- 类别IOU -------")
        for l in range(NUM_CLASSES):
            log_string(f"类别 {seg_label_to_cat[l]:<10} IoU: {IoU[l]:.4f}")

        log_string(f"\n平均IOU: {mean_iou:.4f}")
        log_string(f"整体准确率: {overall_acc:.4f}")
        log_string(f"平均类别准确率: {np.mean(acc_per_class[acc_per_class > 0]):.4f}")

        # 可视化保存
        if args.visual:
            # 保存第一个批次的可视化结果
            with open(os.path.join(visual_dir, f"test_pred.obj"), 'w') as fout, \
                 open(os.path.join(visual_dir, f"test_gt.obj"), 'w') as fout_gt:
                for i in range(min(10000, all_points.shape[0])):  # 限制最大点数
                    pred_color = g_label2color(all_preds[i])
                    gt_color = g_label2color(all_targets[i])
                    fout.write(f"v {all_points[i,0]} {all_points[i,1]} {all_points[i,2]} {pred_color[0]} {pred_color[1]} {pred_color[2]}\n")
                    fout_gt.write(f"v {all_points[i,0]} {all_points[i,1]} {all_points[i,2]} {gt_color[0]} {gt_color[1]} {gt_color[2]}\n")

    log_string("测试完成！")


if __name__ == '__main__':
    args = parse_args()
    main(args)
