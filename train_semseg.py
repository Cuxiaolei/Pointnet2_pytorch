import torch.nn.parallel
import torch.utils.data
import datetime
from pathlib import Path
import numpy as np
import torch
import os
import argparse
import logging
from tqdm import tqdm
import sys
import csv  # 导入csv模块用于保存结果
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

# 计算准确率和IOU的函数
def calculate_metrics(pred, target, num_classes):
    """
    计算每个类别的准确率、IOU和总体准确率
    pred: 预测结果 [B, N]
    target: 目标标签 [B, N]
    num_classes: 类别数量
    """
    # 展平数据
    pred = pred.flatten()
    target = target.flatten()

    # 计算总体准确率
    oa = (pred == target).sum().item() / len(pred) if len(pred) > 0 else 0.0

    # 计算每个类别的TP, FP, FN和准确率
    class_accs = []
    ious = []
    for cls in range(num_classes):
        # 真正例：预测为cls且实际为cls
        tp = ((pred == cls) & (target == cls)).sum().item()
        # 假正例：预测为cls但实际不是cls
        fp = ((pred == cls) & (target != cls)).sum().item()
        # 假负例：实际为cls但预测不是cls
        fn = ((pred != cls) & (target == cls)).sum().item()
        # 总实际为cls的样本数
        total = (target == cls).sum().item()

        # 计算类别准确率
        if total == 0:
            acc = 0.0
        else:
            acc = tp / total
        class_accs.append(acc)

        # 计算IOU，避免除以零
        if tp + fp + fn == 0:
            iou = 0.0
        else:
            iou = tp / (tp + fp + fn)
        ious.append(iou)

    # 计算平均IOU和平均准确率
    miou = sum(ious) / num_classes if num_classes > 0 else 0.0
    macc = sum(class_accs) / num_classes if num_classes > 0 else 0.0

    return oa, class_accs, macc, ious, miou

def main():
    # 设置默认参数
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
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/data/data_s3dis_pointNeXt', help='数据集根目录')
    #
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

    # 创建CSV文件并写入表头
    metrics_file = os.path.join(log_dir, 'metrics.csv')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 表头
        header = ['epoch', 'phase', 'oa', 'macc', 'miou']
        for i in range(NUM_CLASSES):
            header.append(f'class_{i}_acc')
        for i in range(NUM_CLASSES):
            header.append(f'class_{i}_iou')
        writer.writerow(header)

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

    # 统计验证集类别分布
    logger.info("统计验证集类别分布...")
    val_label_counts = np.zeros(NUM_CLASSES, dtype=int)
    for points, labels in val_loader:
        labels_np = labels.numpy().flatten()
        unique, counts = np.unique(labels_np, return_counts=True)
        for u, c in zip(unique, counts):
            val_label_counts[u] += c
    logger.info(f"验证集总类别分布: {dict(zip(range(NUM_CLASSES), val_label_counts))}")

    # 初始化模型
    if args.model == 'pointnet2_sem_seg':
        model = pointnet2_sem_seg.get_model(NUM_CLASSES)
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
    best_val_miou = 0.0  # 用mIOU作为最佳模型指标
    best_model_path = os.path.join(log_dir, "best_model.pth")  # 固定最佳模型路径

    def train(epoch):
        model.train()
        total_loss = 0.0
        # 用于计算指标的累积变量
        all_preds = []
        all_targets = []

        for i, (points, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} 训练")):
            if args.cuda:
                points = points.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            pred, trans_feat = model(points)

            loss = criterion(pred, target, trans_feat)
            loss.backward()
            optimizer.step()

            # 计算预测结果
            pred_choice = pred.data.max(1)[1]
            total_loss += loss.item() * points.size(0)

            # 收集预测结果和目标用于计算指标
            all_preds.append(pred_choice.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            if i % 10 == 0:
                target_np = target.cpu().numpy().flatten()
                unique, counts = np.unique(target_np, return_counts=True)
                batch_dist = dict(zip(unique, counts))
                logger.info(f"训练Batch {i} 类别分布: {batch_dist}")

        # 计算平均损失
        avg_loss = total_loss / len(train_loader.dataset)

        # 计算各项指标
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        oa, class_accs, macc, class_ious, miou = calculate_metrics(all_preds, all_targets, NUM_CLASSES)

        # 日志输出
        logger.info(f"训练 - 轮次: {epoch}")
        logger.info(f"  损失: {avg_loss:.4f}, 总体准确率: {oa:.4f}")
        logger.info(f"  每个类别的准确率: {[f'{acc:.4f}' for acc in class_accs]}")
        logger.info(f"  平均准确率: {macc:.4f}")
        logger.info(f"  每个类别的IOU: {[f'{iou:.4f}' for iou in class_ious]}")
        logger.info(f"  平均IOU: {miou:.4f}")

        # 保存到CSV
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch, 'train', oa, macc, miou]
            row.extend(class_accs)
            row.extend(class_ious)
            writer.writerow(row)

        return avg_loss, oa, macc, miou

    def validate(epoch):
        model.eval()
        total_loss = 0.0
        # 用于计算指标的累积变量
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i, (points, target) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} 验证")):
                if args.cuda:
                    points = points.cuda()
                    target = target.cuda()

                pred, trans_feat = model(points)
                loss = criterion(pred, target, trans_feat)

                # 计算预测结果
                pred_choice = pred.data.max(1)[1]
                total_loss += loss.item() * points.size(0)

                # 收集预测结果和目标
                all_preds.append(pred_choice.cpu().numpy())
                all_targets.append(target.cpu().numpy())

                if i % 10 == 0:
                    target_np = target.cpu().numpy().flatten()
                    unique, counts = np.unique(target_np, return_counts=True)
                    batch_dist = dict(zip(unique, counts))
                    logger.info(f"验证Batch {i} 类别分布: {batch_dist}")

        # 计算平均损失
        avg_loss = total_loss / len(val_loader.dataset)

        # 计算各项指标
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        oa, class_accs, macc, class_ious, miou = calculate_metrics(all_preds, all_targets, NUM_CLASSES)

        # 日志输出
        logger.info(f"验证 - 轮次: {epoch}")
        logger.info(f"  损失: {avg_loss:.4f}, 总体准确率: {oa:.4f}")
        logger.info(f"  每个类别的准确率: {[f'{acc:.4f}' for acc in class_accs]}")
        logger.info(f"  平均准确率: {macc:.4f}")
        logger.info(f"  每个类别的IOU: {[f'{iou:.4f}' for iou in class_ious]}")
        logger.info(f"  平均IOU: {miou:.4f}")

        # 保存到CSV
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch, 'val', oa, macc, miou]
            row.extend(class_accs)
            row.extend(class_ious)
            writer.writerow(row)

        return avg_loss, oa, macc, miou

    # 仅评估模式
    if args.eval:
        logger.info("进入仅评估模式")
        val_loss, val_oa, val_macc, val_miou = validate(start_epoch)
        return

    # 开始训练
    logger.info("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_oa, train_macc, train_miou = train(epoch)

        # 验证
        val_loss, val_oa, val_macc, val_miou = validate(epoch)

        # 学习率调度
        scheduler.step()

        # 保存最佳模型（使用mIOU作为指标）
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            # 保存为固定文件名，会覆盖之前的最佳模型
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"更新最佳模型到 {best_model_path}, 验证mIOU: {best_val_miou:.4f}")

        # 每10轮保存一次模型
        if epoch % 10 == 0:
            save_path = os.path.join(log_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"保存模型到 {save_path}")

    logger.info(f"训练完成，最佳验证mIOU: {best_val_miou:.4f}，最佳模型路径: {best_model_path}")

if __name__ == "__main__":
    main()
