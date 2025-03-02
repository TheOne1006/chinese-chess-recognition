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


- `CChessPackInputs` 
- `CChessRandomFlip`
- `CChessCachedMixUp`
- `CChessHalfFlip`
- `RandomPerspectiveTransform`
- `CChessMixSinglePngCls`
- `CChessCachedCopyHalf`


## todo

- [x] 训练
- [ ] 测试
- [ ] 推理
- [ ] 评估
- [ ] 可视化
- [ ] 部署



