import torch.nn.parallel
import torch.utils.data
import datetime
from pathlib import Path
# 在文件开头添加NumPy导入
import numpy as np
import torch
import os
import argparse
import logging
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# 导入模型和数据集
from models import pointnet2_sem_seg
from data_utils.CustomSemSegDataset import CustomSemSegDataset  # 使用自定义数据集

# 设置日志
def setup_logger(log_file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 计算IOU的函数
def calculate_iou(pred, target, num_classes):
    """
    计算每个类别的IOU和平均IOU
    pred: 预测结果 [B, N]
    target: 目标标签 [B, N]
    num_classes: 类别数量
    """
    # 展平数据
    pred = pred.flatten()
    target = target.flatten()

    # 计算每个类别的TP, FP, FN
    ious = []
    for cls in range(num_classes):
        # 真正例：预测为cls且实际为cls
        tp = ((pred == cls) & (target == cls)).sum().item()
        # 假正例：预测为cls但实际不是cls
        fp = ((pred == cls) & (target != cls)).sum().item()
        # 假负例：实际为cls但预测不是cls
        fn = ((pred != cls) & (target == cls)).sum().item()

        # 计算IOU，避免除以零
        if tp + fp + fn == 0:
            iou = 0.0
        else:
            iou = tp / (tp + fp + fn)
        ious.append(iou)

    # 计算平均IOU
    miou = sum(ious) / num_classes if num_classes > 0 else 0.0

    return ious, miou

def main():
    # 设置默认参数，无需命令行指定
    parser = argparse.ArgumentParser(description='PointNet++ Semantic Segmentation')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='模型名称')
    parser.add_argument('--log_dir', type=str, default='custom_sem_seg_logs', help='日志和模型保存目录')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--npoint', type=int, default=60000, help='每个样本的点数量')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--step_size', type=int, default=20, help='学习率衰减步长')
    parser.add_argument('--gamma', type=float, default=0.5, help='学习率衰减系数')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='不使用CUDA')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--eval', action='store_true', default=False, help='仅评估模式')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    # 请将此处修改为你的数据集实际路径
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/data/data_s3dis_pointNeXt', help='数据集根目录')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # 创建日志目录
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    log_dir = os.path.join(args.log_dir, timestr)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, 'train.log'))
    logger.info(f"参数: {args}")

    # 数据集配置
    NUM_CLASSES = 3  # 3分类任务
    logger.info(f"使用自定义数据集，类别数: {NUM_CLASSES}")

    # 加载数据集
    logger.info("开始加载训练数据...")
    TRAIN_DATASET = CustomSemSegDataset(
        root=args.dataset_root,
        split='train',
        num_point=args.npoint,
        block_size=1.5,
        sample_rate=0.8,
        transform=True
    )

    logger.info("开始加载验证数据...")
    VAL_DATASET = CustomSemSegDataset(
        root=args.dataset_root,
        split='val',
        num_point=args.npoint,
        block_size=1.5,
        sample_rate=1.0,
        transform=False
    )

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.cuda else False,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        VAL_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.cuda else False
    )

    logger.info(f"训练集样本数: {len(TRAIN_DATASET)}, 验证集样本数: {len(VAL_DATASET)}")

    # 加载数据集后，添加验证集类别分布统计
    logger.info("统计验证集类别分布...")
    val_label_counts = np.zeros(NUM_CLASSES, dtype=int)
    for points, labels in val_loader:
        labels_np = labels.numpy().flatten()
        unique, counts = np.unique(labels_np, return_counts=True)
        for u, c in zip(unique, counts):
            val_label_counts[u] += c
    logger.info(f"验证集总类别分布: {dict(zip(range(NUM_CLASSES), val_label_counts))}")

    # 初始化模型
    # 在模型初始化部分修改损失函数的创建方式
    if args.model == 'pointnet2_sem_seg':
        model = pointnet2_sem_seg.get_model(NUM_CLASSES)
        # 使用带权重的损失函数，传入计算好的类别权重
        criterion = pointnet2_sem_seg.get_loss_function(
            weight=torch.FloatTensor(TRAIN_DATASET.label_weights).cuda() if args.cuda else torch.FloatTensor(
                TRAIN_DATASET.label_weights)
        )
    else:
        logger.error(f"未知模型: {args.model}")
        return

    # 移动到GPU
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma
    )

    # 加载预训练模型（如果需要）
    start_epoch = 0
    best_val_miou = 0.0  # 改为用mIOU作为最佳模型指标

    def train(epoch):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_points = 0
        # 用于计算IOU的累积变量
        all_preds = []
        all_targets = []

        for i, (points, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} 训练")):
            if args.cuda:
                points = points.cuda()
                target = target.cuda()  # target形状: [B, N] 其中N是点数量

            optimizer.zero_grad()
            pred, trans_feat = model(points)  # pred形状: [B, C, N] 其中C是类别数

            # 关键修复：调整预测结果维度以适应交叉熵损失
            # F.cross_entropy期望输入形状为 [B*N, C] 或 [B, C, N]
            # 目标标签形状为 [B*N] 或 [B, N]
            # 保持pred为[B, C, N]形状，无需转置为[B, N, C]

            loss = criterion(pred, target, trans_feat)

            loss.backward()
            optimizer.step()

            # 计算准确率
            pred_choice = pred.data.max(1)[1]  # 对于[B, C, N]，在通道维度1上取最大值
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_points += target.size(0) * target.size(1)
            total_loss += loss.item() * points.size(0)

            # 收集预测结果和目标用于计算IOU
            all_preds.append(pred_choice.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            if i % 10 == 0:  # 每10个batch打印一次
                target_np = target.cpu().numpy().flatten()
                unique, counts = np.unique(target_np, return_counts=True)
                batch_dist = dict(zip(unique, counts))
                logger.info(f"训练Batch {i} 类别分布: {batch_dist}")


        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader.dataset)
        avg_acc = total_correct / total_points

        # 计算IOU
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        class_ious, miou = calculate_iou(all_preds, all_targets, NUM_CLASSES)

        # 日志输出
        logger.info(f"训练 - 轮次: {epoch}")
        logger.info(f"  损失: {avg_loss:.4f}, 准确率: {avg_acc:.4f}")
        logger.info(f"  每个类别的IOU: {[f'{iou:.4f}' for iou in class_ious]}")
        logger.info(f"  平均IOU: {miou:.4f}")

        return avg_loss, avg_acc, miou

    # 同时修改验证函数
    def validate(epoch):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_points = 0
        # 用于计算IOU的累积变量
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i, (points, target) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} 验证")):
                if args.cuda:
                    points = points.cuda()
                    target = target.cuda()

                pred, trans_feat = model(points)  # [B, C, N]

                # 保持pred为[B, C, N]形状，不转置
                loss = criterion(pred, target, trans_feat)

                # 计算准确率 - 在通道维度1上取最大值
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                total_correct += correct.item()
                total_points += target.size(0) * target.size(1)
                total_loss += loss.item() * points.size(0)

                # 收集预测结果和目标
                all_preds.append(pred_choice.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        # 计算平均损失和准确率
        avg_loss = total_loss / len(val_loader.dataset)
        avg_acc = total_correct / total_points

        # 计算IOU
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        class_ious, miou = calculate_iou(all_preds, all_targets, NUM_CLASSES)

        # 日志输出
        logger.info(f"验证 - 轮次: {epoch}")
        logger.info(f"  损失: {avg_loss:.4f}, 准确率: {avg_acc:.4f}")
        logger.info(f"  每个类别的IOU: {[f'{iou:.4f}' for iou in class_ious]}")
        logger.info(f"  平均IOU: {miou:.4f}")

        if i % 10 == 0:
            target_np = target.cpu().numpy().flatten()
            unique, counts = np.unique(target_np, return_counts=True)
            batch_dist = dict(zip(unique, counts))
            logger.info(f"验证Batch {i} 类别分布: {batch_dist}")


        return avg_loss, avg_acc, miou

    # 仅评估模式
    if args.eval:
        logger.info("进入仅评估模式")
        val_loss, val_acc, val_miou = validate(start_epoch)
        return

    # 开始训练
    logger.info("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_acc, train_miou = train(epoch)

        # 验证
        val_loss, val_acc, val_miou = validate(epoch)

        # 学习率调度
        scheduler.step()

        # 保存最佳模型（使用mIOU作为指标）
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            save_path = os.path.join(log_dir, f"best_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"保存最佳模型到 {save_path}, 验证mIOU: {best_val_miou:.4f}")

        # 每10轮保存一次模型
        if epoch % 10 == 0:
            save_path = os.path.join(log_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"保存模型到 {save_path}")

    logger.info(f"训练完成，最佳验证mIOU: {best_val_miou:.4f}")

if __name__ == "__main__":
    main()
