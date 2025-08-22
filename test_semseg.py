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
classes = ['class0', 'class1', 'class2']  # 替换为实际类别名称
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


def parse_args():
    parser = argparse.ArgumentParser('PointNet++ Semantic Segmentation Testing')
    # 与训练代码同步的默认参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备')
    parser.add_argument('--num_point', type=int, default=8192, help='每个样本的点数量')
    parser.add_argument('--log_dir', type=str, default='custom_sem_seg_logs', help='日志和模型根目录')
    parser.add_argument('--visual', action='store_true', default=False, help='是否可视化结果')
    parser.add_argument('--num_votes', type=int, default=3, help='投票聚合次数')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='不使用CUDA')
    # 数据集路径（与训练代码默认路径一致）
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/data/data_s3dis_pointNeXt', help='数据集根目录')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    """累积投票结果"""
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


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

        log_string('---- 开始全场景评估 ----')

        # 遍历测试数据
        for scene_idx in range(len(TEST_DATASET)):
            scene_name = f"scene_{scene_idx}"
            log_string(f"处理场景 [{scene_idx + 1}/{len(TEST_DATASET)}]: {scene_name}")

            # 获取完整场景数据（假设CustomSemSegDataset实现了这些方法）
            whole_scene_data = TEST_DATASET.get_whole_scene(scene_idx)
            whole_scene_label = TEST_DATASET.get_whole_scene_labels(scene_idx)
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))

            # 多轮投票
            for _ in tqdm(range(args.num_votes), desc="投票轮次"):
                # 获取场景的分块数据
                scene_blocks, scene_block_labels, scene_block_smpw, scene_block_indices = TEST_DATASET.get_scene_blocks(scene_idx)
                num_blocks = len(scene_blocks)
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE

                # 批次处理
                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx

                    # 准备批次数据
                    batch_data = np.array(scene_blocks[start_idx:end_idx])
                    batch_indices = np.array(scene_block_indices[start_idx:end_idx])
                    batch_smpw = np.array(scene_block_smpw[start_idx:end_idx])

                    # 转换为tensor并移动到设备
                    torch_data = torch.FloatTensor(batch_data)
                    if args.cuda:
                        torch_data = torch_data.cuda()
                    torch_data = torch_data.transpose(2, 1)  # [B, C, N]，与训练代码输入格式一致

                    # 模型推理（与训练代码输出格式匹配）
                    seg_pred, _ = model(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(1)[1].numpy()  # 在类别维度取最大值

                    # 累积投票
                    vote_label_pool = add_vote(
                        vote_label_pool,
                        batch_indices,
                        batch_pred_label,
                        batch_smpw
                    )

            # 确定最终预测标签
            pred_label = np.argmax(vote_label_pool, 1)

            # 计算每类指标
            for l in range(NUM_CLASSES):
                total_seen = np.sum(whole_scene_label == l)
                total_correct = np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno = np.sum((pred_label == l) | (whole_scene_label == l))

                total_seen_class[l] += total_seen
                total_correct_class[l] += total_correct
                total_iou_deno_class[l] += total_iou_deno

            # 计算当前场景的mIOU
            iou_map = np.array([
                total_correct_class[l] / (total_iou_deno_class[l] + 1e-6)
                for l in range(NUM_CLASSES)
            ])
            valid_classes = [l for l in range(NUM_CLASSES) if total_seen_class[l] > 0]
            scene_miou = np.mean(iou_map[valid_classes]) if valid_classes else 0.0
            log_string(f"场景 {scene_name} 平均IOU: {scene_miou:.4f}")

            # 可视化保存
            if args.visual:
                with open(os.path.join(visual_dir, f"{scene_name}_pred.obj"), 'w') as fout, \
                     open(os.path.join(visual_dir, f"{scene_name}_gt.obj"), 'w') as fout_gt:
                    for i in range(whole_scene_data.shape[0]):
                        pred_color = g_label2color(pred_label[i])
                        gt_color = g_label2color(whole_scene_label[i])
                        fout.write(f"v {whole_scene_data[i,0]} {whole_scene_data[i,1]} {whole_scene_data[i,2]} {pred_color[0]} {pred_color[1]} {pred_color[2]}\n")
                        fout_gt.write(f"v {whole_scene_data[i,0]} {whole_scene_data[i,1]} {whole_scene_data[i,2]} {gt_color[0]} {gt_color[1]} {gt_color[2]}\n")
                with open(os.path.join(visual_dir, f"{scene_name}_pred.txt"), 'w') as f:
                    for label in pred_label:
                        f.write(f"{label}\n")

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

    log_string("测试完成！")


if __name__ == '__main__':
    args = parse_args()
    main(args)
