import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 导入必要的组件
from mmengine.config import Config
from cchess_reg.apis.cchess_image_classification import CChessImageClassificationInferencer

from cchess_reg.models import *
from cchess_reg.datasets import *

def parse_args():
    parser = argparse.ArgumentParser(description='中国象棋棋盘识别演示')
    parser.add_argument('img', help='输入图像路径')
    parser.add_argument(
        '--config', 
        default='configs/swinv2-nano_cchess16-256.py',
        help='配置文件路径')
    parser.add_argument(
        '--checkpoint', 
        default='checkpoints/epoch_200.pth',
        help='checkpoint文件路径')
    parser.add_argument(
        '--device', 
        default='cuda:0', 
        help='推理设备，可选：cuda:0, cpu')
    parser.add_argument(
        '--score-thr', 
        type=float, 
        default=0.5, 
        help='识别阈值')
    parser.add_argument(
        '--out-file', 
        default=None, 
        help='输出结果图像路径')
    return parser.parse_args()


def visualize_chess_board(img, results, score_thr=0.5, out_file=None):
    """可视化棋盘识别结果"""
    plt.figure(figsize=(12, 10))
    
    # 显示原始图像
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 获取图像尺寸
    h, w = img.shape[:2]
    
    # 棋盘有10行9列
    rows, cols = 10, 9
    
    offset_x, offset_y = 50, 50
    
    # 计算每个格子的大小
    cell_h, cell_w = (h - offset_y * 2) / (rows - 1), (w - offset_x * 2) / (cols - 1)
    
    # 解析结果
    pred_scores = results[0]['pred_scores']  # 形状为 [90, 16]
    pred_labels = results[0]['pred_label']   # 形状为 [90]
    pred_classes = results[0]['pred_class']  # 形状为 [90]
    pred_score = results[0]['pred_score']    # 形状为 [90]
    
    # 绘制棋盘格子和标签
    for i in range(rows * cols):
        row, col = i // cols, i % cols
        
        # 只显示置信度大于阈值的预测
        if pred_score[i] > score_thr:
            # 计算格子左上角坐标
            x, y = col * cell_w + offset_x - cell_w/2, row * cell_h + offset_y - cell_h/2
            
            # 绘制矩形和标签
            rect = Rectangle((x, y), cell_w, cell_h, 
                             linewidth=1, edgecolor='r', facecolor='none', alpha=0.5)
            plt.gca().add_patch(rect)
            
            # 获取类别和置信度
            label = pred_classes[i]
            conf = pred_score[i]
            
            # 根据标签是否为大写选择不同的颜色
            if label.isupper():
                facecolor = 'RED'
            else:
                facecolor = 'green'
                
            if label == 'x':
                facecolor = 'black'
            elif label == '.':
                facecolor = 'white'
                
            # 在格子中心绘制类别和置信度
            plt.text(x + cell_w/2, y + cell_h/2, 
                     f'{label}\n{conf:.2f}',
                     color='white', 
                     bbox=dict(facecolor=facecolor, alpha=0.5),
                     ha='center', va='center')
    
    plt.title("cchess reg")
    plt.axis('off')
    
    if out_file:
        plt.savefig(out_file, bbox_inches='tight')
        print(f'结果已保存至 {out_file}')
    
    plt.show()


def format_fen(results, score_thr=0.5):
    """将识别结果转换为FEN格式"""
    pred_classes = results[0]['pred_class']  # 形状为 [90]
    pred_score = results[0]['pred_score']    # 形状为 [90]
    
    # 初始化FEN字符串
    fen = []
    
    # 遍历每一行
    for row in range(10):
        row_fen = []
        empty_count = 0
        
        for col in range(9):
            idx = row * 9 + col
            piece = pred_classes[idx]
            confidence = pred_score[idx]
            
            # 只处理置信度大于阈值的预测
            if confidence > score_thr:
                if piece == '.' or piece == 'x':  # 空白点
                    empty_count += 1
                else:  # 棋子
                    if empty_count > 0:
                        row_fen.append(str(empty_count))
                        empty_count = 0
                    row_fen.append(piece)
            else:  # 置信度低，视为空白
                empty_count += 1
        
        # 处理行尾的空白
        if empty_count > 0:
            row_fen.append(str(empty_count))
        
        # 将当前行加入FEN
        fen.append(''.join(row_fen))
    
    # 按照FEN格式连接，使用'/'分隔行
    return '/'.join(fen)


def main():
    args = parse_args()
    
    # 加载图像
    img = cv2.imread(args.img)
    if img is None:
        print(f'无法加载图像：{args.img}')
        return
    
    # 初始化推理器
    inferencer = CChessImageClassificationInferencer(
        model=args.config,
        pretrained=args.checkpoint,
        device=args.device
    )
    
    # 进行推理
    results = inferencer(img)
    
    # 可视化结果
    visualize_chess_board(img, results, args.score_thr, args.out_file)
    
    # 生成并显示FEN表示
    fen = format_fen(results, args.score_thr)
    print("FEN表示：")
    print(fen)


if __name__ == '__main__':
    # python examples/inference_demo.py data/cchess_multi_label_layout/val/20250219_qilu_val_250218123431.jpg  --device cpu --out-file examples/work_dirs/demo.jpg
    main()
