# 中国象棋识别

- Chinese Chess Recognition 
- Xiangqi Recognition

基于 cnn 的中国象棋、棋谱识别, 欢迎 讨论 交流。
  

## 在线 Demo
[huggingface space demo](https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition)


## 相关技术

- MMPose
- MMDetection
- MMPreTrain


## 数据集采集

- 30 多个象棋视频


> Tips: 
>> 1. 数据集中 使用 "." 表示空白区域， 象棋拥有 32 棋子，分布在 90 个网格中，没有棋子的地方，使用 "." 表示
>> 2. 数据集中 使用 “x” 表示 异常元素，即不属于棋子，也不属于棋盘网格


## 流程

1. 棋盘检测(可缺省)
2. 关键点检测
3. 透视变换
4. 棋子识别



## 棋盘 Bounding Box Detection

由于 keypoint detection, 选择 topdown 的方式，可以减少棋盘检测的难度, 以及减少 数据采集的工作量


## 关键点检测

目前采用 topdown 的方式, 预测 34 个关键点信息。用于透视变换。
将棋盘转换为俯视图, 用于棋子识别

<img src="assets/keypoints.png" alt="keypoints" style="max-height: 500px;" />



#### 透视变换

通过 4 个 角，和 将、帅 的位置，进行透视变换，将棋盘转换为俯视图

<img src="assets/perspective.webp" alt="perspective" style="max-height: 500px;" />


#### !tips: 俯视图与原图识别

> 优点:
>> 1. 减少难度，模型可以设计的更简单，不用考虑棋盘的旋转，棋局的分布
>> 2. 输入尺寸单一，比例固定，模型更容易收敛
>> 3. 标注更简单、方便

> 缺点:
>> 1. 透视变换精度要求较高，要去 keypoint detection 精度较高, 否则识别效果较差
    1.1 如果 四个角 与实际角度偏差大，可能导致 透视变换后，棋盘变形严重，导致识别效果较差
>> 2. 俯视图棋子密集，容易识别错误
    2.1 拍摄角度与棋盘水平面过于接近，网格交点容易被棋子覆盖遮挡，增加识别难度
    2.2 废弃棋子，可能叠加在其他棋子上，导致识别时，进入棋盘区域中，造成识别错误


eg: 

<img src="assets/bad_shooting_angle2.jpg" alt="bad_shooting_angle" style="max-height: 150px;" />

<img src="assets/bad_shooting_angle2_transformed.png" alt="bad_shooting_angle_transformed" style="max-height: 150px;" />


如上图，拍摄角度与棋盘水平面过于接近，黑炮后面的网格交点被棋子覆盖，容易导致识别错误，即使透视变换成俯视图，也会因为棋子变形而遮盖掉该点的信息.



## 棋子识别

- 采用 resnet50 作为 backbone
- 直接使用 卷积 来处理 特征信息，将 2048 维的特征，降维到 16 维度，最终特征宽高为 10x9, 正好符合 10x9 的棋盘
- 再将俯视图 调整到 固定尺寸，送入 分类网络中，进行识别


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

10 行 9 列，16 个类别


#### 为什么不用分割成块？

<img src="assets/detected_lines.png" alt="detected_lines" style="max-height: 250px;" />

1. 类似国际象棋的方法，具有一定局限性，
2. 既要考虑拍摄角度，不能过于倾斜，又要求棋子摆放整齐，显然在实际过程中，很难满足

## Evaluation

1. 可以使用 multi-label 的分类方式，进行评估
   -  但是由于 数据类型，分布相当不均匀，导致评估结果和实际效果相差较大，
   -  在评估时，可以增加一种，去除 “.” 的评估方式，提高评估准确率
2. 使用 10x9 的特征图，进行识别
3. 由于 数据分类的分布，可能会根据 数据集的变化，所以我额外，增加了，对每个 分类的 precision, recall, f1-score 的评估信息，用以更直观看到，每个分类的识别效果


## 如何提高识别准确率

1. 提高透视变换的精度
2. 使用 象棋的规则，辅助识别
   1. 例如 一方的相，只可能出现在 7 个位置，
   2. 例如 将、士 只能出现在 指定的 3x3 的区域中
3. 采集更多，棋子、棋盘的图片，提高模型泛化能力


## TODO:

- [ ] keypoint detection 使用 bottomup 的方式, 移除 BBox 的检测
- [ ] 使用 6 个关键点, 减少计算量
- [ ] 兼容更多棋盘样式
- [ ] 棋子识别时，增加与周围棋子的关联性，提高识别准确率
- [ ] 视频追踪
- [ ] 开放配置信息以及扩展代码
- [ ] 数据标注方案

## 参考

- [End-to-End Chess Recognition](https://arxiv.org/html/2310.04086)


