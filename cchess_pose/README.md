# CChess Pose Estimation

Base on [MMPose](https://github.com/open-mmlab/mmpose)


## Algorithm

- 4 个角的关键点，相对简单
- 而且希望模型尽可能小，所以选择 RTMPose

### 其他 Algorithm 尝试

- [yoloxpose_tiny](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#yolo-pose-cvprw-2022)



## 目录结构

```
cchess_pose/
├── configs/
├── data/
├── datasets/
├── models/
|   |-- heads/rtmcc_head2.py
├── train.py
├── test.py
```

1. rtmcc_head2 继承 `RTMHead`, 扩展了 RTMHead 对其他 loss 的支持
2. 数据集为 coco 模式


### datasets


```
datasets/
├── datasets/ # 自定义Dataset 替代默认 Dataset
├── transforms/ # 自定义Transforms
├── __init__.py
```


### transforms

- `CChessRandomFlip4` 随机翻转
- `RandomPerspectiveTransform` 随机透视变换
- `RandomUseFullImg` 随机使用完整图片
- `CopyParseWithPose4` 根据 4 个角点，替换为其他棋盘的数据
- `RandomGetBBoxCenterScale` 随机获取bbox中心和缩放
- `RandomHalfCChess4` 随机镜像翻转半个棋盘



## Scripts

```bash
cd cchess_pose
# 训练
python tools/train.py configs/rtmpose-4/rtmpose-t-cchess_4.py

```



## TODO

- [x] 训练
  - [ ] 多卡训练
- [ ] 测试
- [ ] 推理
- [ ] 部署


## Dataset

