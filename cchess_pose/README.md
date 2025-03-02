# CChess Pose Estimation

Base on [MMPose](https://github.com/open-mmlab/mmpose)


## Algorithm

- 4 个角的关键点，相对简单
- 而且希望模型尽可能小，所以选择 RTMPose

### 其他 Algorithm 尝试

- [yoloxpose_tiny](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#yolo-pose-cvprw-2022)



## 目录


## Scripts

```bash
cd cchess_pose
# 训练
python train.py configs/rtmpose-4/rtmpose-t-cchess_4.py

```



## TODO

- [x] 训练
  - [ ] 多卡训练
- [ ] 测试
- [ ] 推理
- [ ] 部署


## Dataset

