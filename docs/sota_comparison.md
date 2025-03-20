# 中国象棋识别系统与SOTA对比分析

目前仅找到 棋路 App 与 天天象棋，两个 App 存在识别功能
- App 目前 没有相关数据
- 以下都是自己收集并整的数据


> 功能对比 

|功能| cchess_reg| 天天象棋 | 棋路  |
|--|--|--|--|
|自动对齐| ✅ | ❌ | ❌ |
|抗干扰| ✅ | ❌ | ❌ |



## 与SOTA对比分析

- 由于 App 对少对齐功能
- 为了统一对比，使用第一产生的图片， 再进行 app 识图



## N 张图对比

共有 17 组对比图片：

| 编号 | 原始图片 | 自动对齐 | cchess_reg | 天天象棋 | 棋路 |
|------|---------|---------|------------|---------|------|
| 1 | ![原始图片1](/docs/sota/origin/1.png) | ![对齐图片1](/docs/sota/alignment/1.png) | ![cchess_reg1](/docs/sota/cchess_reg/1.png) | ![天天象棋1](/docs/sota/ttxq/1.png) | ![棋路1](/docs/sota/ql/1.png) |
| 2 | ![原始图片2](/docs/sota/origin/2.png) | ![对齐图片2](/docs/sota/alignment/2.png) | ![cchess_reg2](/docs/sota/cchess_reg/2.png) | ![天天象棋2](/docs/sota/ttxq/2.png) | ![棋路2](/docs/sota/ql/2.jpg) |
| 3 | ![原始图片3](/docs/sota/origin/3.png) | ![对齐图片3](/docs/sota/alignment/3.png) | ![cchess_reg3](/docs/sota/cchess_reg/3.png) | ![天天象棋3](/docs/sota/ttxq/3.png) | ![棋路3](/docs/sota/ql/3.jpg) |
| 4 | ![原始图片4](/docs/sota/origin/4.png) | ![对齐图片4](/docs/sota/alignment/4.png) | ![cchess_reg4](/docs/sota/cchess_reg/4.png) | ![天天象棋4](/docs/sota/ttxq/4.png) | ![棋路4](/docs/sota/ql/4.jpg) |
| 5 | ![原始图片5](/docs/sota/origin/5.png) | ![对齐图片5](/docs/sota/alignment/5.png) | ![cchess_reg5](/docs/sota/cchess_reg/5.png) | ![天天象棋5](/docs/sota/ttxq/5.png) | ![棋路5](/docs/sota/ql/5.jpg) |
| 6 | ![原始图片6](/docs/sota/origin/6.png) | ![对齐图片6](/docs/sota/alignment/6.png) | ![cchess_reg6](/docs/sota/cchess_reg/6.png) | ![天天象棋6](/docs/sota/ttxq/6.png) | ![棋路6](/docs/sota/ql/6.jpg) |
| 7 | ![原始图片7](/docs/sota/origin/7.png) | ![对齐图片7](/docs/sota/alignment/7.png) | ![cchess_reg7](/docs/sota/cchess_reg/7.png) | ![天天象棋7](/docs/sota/ttxq/7.png) | ![棋路7](/docs/sota/ql/7.jpg) |
| 8 | ![原始图片8](/docs/sota/origin/8.png) | ![对齐图片8](/docs/sota/alignment/8.png) | ![cchess_reg8](/docs/sota/cchess_reg/8.png) | ![天天象棋8](/docs/sota/ttxq/8.png) | ![棋路8](/docs/sota/ql/8.jpg) |
| 9 | ![原始图片9](/docs/sota/origin/9.png) | ![对齐图片9](/docs/sota/alignment/9.png) | ![cchess_reg9](/docs/sota/cchess_reg/9.png) | ![天天象棋9](/docs/sota/ttxq/9.png) | ![棋路9](/docs/sota/ql/9.jpg) |
| 10 | ![原始图片10](/docs/sota/origin/10.png) | ![对齐图片10](/docs/sota/alignment/10.png) | ![cchess_reg10](/docs/sota/cchess_reg/10.png) | ![天天象棋10](/docs/sota/ttxq/10.png) | ![棋路10](/docs/sota/ql/10.jpg) |
| 11 | ![原始图片11](/docs/sota/origin/11.png) | ![对齐图片11](/docs/sota/alignment/11.png) | ![cchess_reg11](/docs/sota/cchess_reg/11.png) | ![天天象棋11](/docs/sota/ttxq/11.png) | ![棋路11](/docs/sota/ql/11.jpg) |
| 12 | ![原始图片12](/docs/sota/origin/12.png) | ![对齐图片12](/docs/sota/alignment/12.png) | ![cchess_reg12](/docs/sota/cchess_reg/12.png) | ![天天象棋12](/docs/sota/ttxq/12.png) | ![棋路12](/docs/sota/ql/12.jpg) |
| 13 | ![原始图片13](/docs/sota/origin/13.png) | ![对齐图片13](/docs/sota/alignment/13.png) | ![cchess_reg13](/docs/sota/cchess_reg/13.png) | ![天天象棋13](/docs/sota/ttxq/13.png) | ![棋路13](/docs/sota/ql/13.jpg) |
| 14 | ![原始图片14](/docs/sota/origin/14.png) | ![对齐图片14](/docs/sota/alignment/14.png) | ![cchess_reg14](/docs/sota/cchess_reg/14.png) | ![天天象棋14](/docs/sota/ttxq/14.png) | ![棋路14](/docs/sota/ql/14.jpg) |
| 15 | ![原始图片15](/docs/sota/origin/15.png) | ![对齐图片15](/docs/sota/alignment/15.png) | ![cchess_reg15](/docs/sota/cchess_reg/15.png) | ![天天象棋15](/docs/sota/ttxq/15.png) | ![棋路15](/docs/sota/ql/15.jpg) |
| 16 | ![原始图片16](/docs/sota/origin/16.png) | ![对齐图片16](/docs/sota/alignment/16.png) | ![cchess_reg16](/docs/sota/cchess_reg/16.png) | ![天天象棋16](/docs/sota/ttxq/16.png) | ![棋路16](/docs/sota/ql/16.jpg) |
| 17 | ![原始图片17](/docs/sota/origin/17.webp) | ![对齐图片17](/docs/sota/alignment/17.png) | ![cchess_reg17](/docs/sota/cchess_reg/17.png) | ![天天象棋17](/docs/sota/ttxq/17.png) | ![棋路17](/docs/sota/ql/17.jpg) |




## 正确率统计

各系统棋子识别错误数量

| 编号 | cchess_reg | 天天象棋 | 棋路 |
|------|------------|---------|------|
| 1   |   0 |   1 | 0 |
| 2   |   0 |   0   | 0 |
| 3  |  0 | ❌ |  ❌
| 4 | 0 | 0 |  0 |
| 5 | 2 | ❌ | 5 |
| 6 | 2  | ❌ |  ❌ |
| 7 | 0 | ❌ |  ❌ |
| 8 | 1 | ❌ |  ❌ |
| 9 | 0 |  1 |  0|
| 10 | 3  | 4   | 6|
| 11 | 0 | 0 | 0 |
| 12 | 0 | 0  | 0 |
| 13 | 0 | 0  | 0 |
| 14 | 0 | 0  | 8 |
| 15 | 0  |  0 | 0 |
| 16 | 0 | 0 | 0 |
| 17 | 0 | 0 | 0 |


> tips:
> ❌: 差异太多，或者识别失败


## 优势分析

### cchess_reg系统优势
- 拍摄角度可以支持更大范围
- 对非棋子有独立分类
- 更高的鲁棒性


