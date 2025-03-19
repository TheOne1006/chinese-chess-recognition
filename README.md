# 中国象棋识别、棋谱识别

- Chinese Chess Recognition 
- Xiangqi Recognition

# 在线 Demo

[huggingface space demo](https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition)

> 黑子(将士相马车炮卒 对应简写`kabnrcp`) 、红子(帅将相马车炮卒 对应简写`KABNRCP`) +  其他 简写`x`  + 空位 简写`.`



[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheOne1006/chinese-chess-recognition/blob/main/cchess_pose/examples/run_on_colab.ipynb) pose-detection 


[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheOne1006/chinese-chess-recognition/blob/main/cchess_reg/examples/run_on_colab.ipynb) cchess_reg 



## Pretrained Model

- https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition/tree/main/onnx



# 摘要

真实场景下的象棋识别。能够直接从图像中预测棋局信息(棋盘位置，棋子位置，棋子类型)。    

在真实场景下，拍摄角度、棋盘、棋子都会影响识别效果。  
本项目尝试使用深度学习 并通过两步进行 **棋盘关键点检测**、**多区域分类检测**。

并采用 Swin Transformer2 网络架构，将最终生成特征图 10x9x16，来预测棋局信息。


![Figure 1](assets/flow.jpg)


# 背景


日常场景的棋局识别时，不仅需要考虑 棋子的位置和分类，还需要考虑 棋盘的倾斜、棋子摆放不正、棋盘背景复杂等场景。



# RELATED WORK

- 分步处理策略
  - 尝试方法
  - 端到端识别
  - 物体检测与关键点检测
  - 多区域分类检测

## 分步处理策略

### 其他方法的尝试

项目开始时，尝试了多种方法，但都存在局限性。


比如：  
1. 基于文字识别，效果并不理想。
2. YOLO, 需要不同角度，不同大小的棋子，需要大量多样的数据。
3. 使用 LineNet 模型，通过 棋盘的直线检测。
  - ![Figure 3](assets/detected_lines.png)
  - 但象棋的棋子是在线上的，会干扰直线的预测


### 端到端识别

这是最终的目的?
但这也是最困难最复杂的方案，需要足够多的数据资源。
参考国际象棋领域的 [End-to-End Chess Recognition](https://arxiv.org/html/2310.04086?_immersive_translate_auto_translate=1)，
基于该文章的思路，可以实现。
真实场景的标注样本也很难获取和标注。


综上，分步处理对我来说，是更可行的方案，每一步都是相对简单，更容易验证。 


### 关键点检测

34 个关键点，棋盘的外圈。标注为

```
A0 A1 A2 A3 A4 A5 A6 A7 A8
B0                      B8
C0                      C8
D0                      D8
...
E-I
...
J0 J1 J2 J3 J4 J5 J6 J7 J8
```

最终保留了 A0、A8、J0、J8 四个关键点。

- A0 与 A8, 为 黑方 的两个角点
- J0 与 J8, 为 红方 的两个角点


### 多区域分类检测

| 前置: 透视变化, 将图片转换成统一格式

通过 4 个角点，可以将图片透视变化成俯视图。  

该方法非常契合中国象棋: 

1. 可以减少棋盘倾斜的影响: 全部转换成俯视角度，棋盘倾斜的影响基本可以忽略。（国际象棋俯视图就不适用)
2. 可以减少背景干扰。
3. 棋盘、棋子大小不一的影响: 透视变化后， 基本使得棋盘大小一致，棋子大小也会在一定范围内。


| 多区域

10x9 的区域，每个区域 16 分类。耦合象棋布局。

将俯视图 通过 backbone 后，再使用 卷积 来处理特征信息，最终特征宽高为 10x9x16

最终输出特征图

```
          +----------+
         /          /|
        /         16 |
       /          /  |
      +----9-----+   |
      |          |   |
      |          |   |
      |          |   |
      |         10   +
      |          |  /
      |          | /
      |          |/
      +----------+
```


### 16 分类

16分类包含: 黑子(将士相马车炮卒 对应简写`kabnrcp`) 、红子(帅将相马车炮卒 对应简写`KABNRCP`) +  其他 简写`x`  + 空位 简写`.`


# 数据集与评估

- 数据集
- 数据增强
- 损失函数
- 评估指标

### 数据集

pose demo: https://drive.google.com/file/d/1pbUt9mTVxpNQahZUmYKMPYMh0MS7KJ3O/view?usp=drive_link


cchess_reg demo: https://drive.google.com/file/d/1EHv_pafyCb6qa0DTwZVL3yrz4wHt7TuM/view?usp=drive_link

### 数据增强

除了常规的方法，还增加了一些针对象棋的增强:

1. `CChessMixSinglePngCls`: 填充棋子到棋盘上，并修改对应 label, 并对棋子进行随机旋转。
2. `CChessHalfFlip`: 象棋版本的翻转。

### 损失函数

loss 使用 `KLDiscretLoss` + `SmoothL1Loss`

todo:
1. `Wing Loss`
2. `FocalLoss`


### 评估指标

除了传统的多分类方法，(Accuracy、Recall、F1)
但差异不大，

- 使用整局90个棋子全部正确，视为完全正确
  - full-err-k: $= \frac{\text{出现的错误} \leq k \text{ 个棋子的棋局}}{\text{所有棋局}}$
- 在本校验集中
  - full-err-zero: 0.83
  - full-err-1: 0.90
  - full-err-3: 0.96


# TODO
- 局限性
  - 黑车 与 黑卒，相似度高
  - 角度，俯视最佳，倾斜角度也会影响识别率
- 更多方向
  - 收集数据(更多棋子样式，棋盘样式)

# 技术框架

本项目基于以下开源框架开发：

- [MMPose](https://github.com/open-mmlab/mmpose)：用于棋盘关键点检测
- [MMPretrain](https://github.com/open-mmlab/mmpretrain)：用于棋子分类识别

感谢OpenMMLab团队提供的这些优秀工具，使本项目的开发成为可能。


# 参考文献


# 附录

- [End-to-End Chess Recognition](https://arxiv.org/html/2310.04086?_immersive_translate_auto_translate=1)
- [RTMPose]()


## 源码

https://github.com/TheOne1006/chinese-chess-recognition


安装

```bash
pip install -e .
```

文件结构

```
.
├── cchess_pose/  # 关键点检测
├── cchess_reg/  # 棋局分类
```


## 数据集

- 数据统计

- 示例图像

![result](assets/demo001_result.png)
