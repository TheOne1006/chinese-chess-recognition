# 棋盘检测(多区域单分类)



## 目录结构

```
cchess_reg/
├── configs/
├── data/
├── models/
├── datasets/
├── evaluation/
├── structures/
├── train.py
├── test.py
├── README.md
```



## 数据集


### 数据集格式

`txt` 为原始标注文件，
`annotation/train.json` 为 coco 化后的标注文件

```
data/
├── cchess_multi_label_layout/
│   ├── train/
│   │   ├── xxxx.jpg
│   │   └── xxxx.txt
│   ├── val/
│   │   ├── xxxx.jpg
│   │   └── xxxx.txt
│   └── annotations/
│       ├── train.json
│       └── val.json
```

### 数据集处理

`datasets/transforms` 为扩展数据处理，以及数据增强


- `CChessPackInputs` 替换原生 `PackInputs`
- `CChessRandomFlip` 替换原生 `RandomFlip` 随机翻转
- `CChessCachedMixUp` 替换原生 `CachedMixUp` 随机混合，Deprecated, 使用 `CChessMixSinglePngCls` 替代
- `CChessHalfFlip` 镜像
- `RandomPerspectiveTransform` 随机透视变换
- `CChessMixSinglePngCls` 增加棋谱的棋子，多样性、以及处理分类均衡问题
- `CChessCachedCopyHalf` 随机替换 上半部 或者 下半部 棋谱


## todo

- [x] 训练
- [ ] 测试
- [ ] 推理
- [ ] 评估
- [ ] 可视化
- [ ] 部署


## evaluation

增加对应的 16 分类的 评估指标，以及 90 个位置 的相关评估

## Models

1. 添加 Chess10_9Neck 将, feature 降维 10 * 9 * 16
2. `CChessTableHead` 继承 `MultiLabelClsHead`
   - 将输出一个多分类的结果，改为固定 90 个位置的分类结果
   - 以及相关损失函数的计算逻辑修改
  
## structures

1. `CChessDataSample` 继承 `BaseDataSample` 
   - 增加 `pred_score` 和 `gt_score` 用于存储预测得分和真实得分
   - 增加 `pred_label` 和 `gt_label` 用于存储预测标签和真实标签


## Script


```bash
# 训练
python train.py configs/swinv2-nano_cchess16-256.py

```
