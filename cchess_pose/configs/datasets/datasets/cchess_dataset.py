from mmpose.datasets.datasets.base import BaseCocoStyleDataset
from mmpose.registry import DATASETS

def generate_max_gap_colors(n=4):
    colors = []
    for i in range(n):
        # 使用HSV色彩空间，H值均匀分布在0-360度之间
        hue = i * (360 / n)
        # 固定饱和度和明度为最大
        saturation = 100
        value = 100
        
        # 将HSV转换为RGB（0-255范围）
        h = hue / 360
        s = saturation / 100
        v = value / 100
        
        if s == 0:
            r = g = b = v
        else:
            h *= 6
            i = int(h)
            f = h - i
            p = v * (1 - s)
            q = v * (1 - s * f)
            t = v * (1 - s * (1 - f))
            
            if i == 0:
                r, g, b = v, t, p
            elif i == 1:
                r, g, b = q, v, p
            elif i == 2:
                r, g, b = p, v, t
            elif i == 3:
                r, g, b = p, q, v
            elif i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
        
        # 转换到0-255范围
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    
    return colors

_bone_names = [
    "A0", "A8",
    "J0", "J8",
]

_skeleton_links = [
    "A0-A8",
    "J0-J8",
    "A0-J0", "A8-J8",
]

# 生成4个颜色
all_keypoint_colors = generate_max_gap_colors(len(_bone_names))
all_skeleton_colors = generate_max_gap_colors(len(_skeleton_links))


keypoint_info_array = []

for id, bone_name in enumerate(_bone_names):
    # type 为 upper 或 lower

    if bone_name in ['A0', 'A8']:
        type = 'lower'
    else:
        type = 'upper'

    keypoint_info_array.append(dict(
            name=bone_name,
            id=id,
            color=all_keypoint_colors[id],
            type=type,
            swap=''))
    

_skeleton_info_array = []

for id, skeleton_link in enumerate(_skeleton_links):

    start_bone_name, end_bone_name = skeleton_link.split('-')
    _skeleton_info_array.append(dict(
            link=(start_bone_name, end_bone_name),
            id=id,
            color=all_skeleton_colors[id],
            ))

_joint_weights = [1.0] * len(_bone_names)

_keypoint_info = {
    keypoint_info['id']: keypoint_info for keypoint_info in keypoint_info_array
}

_skeleton_info = {
    skeleton_info['id']: skeleton_info for skeleton_info in _skeleton_info_array
}

dataset_info = dict(
    dataset_name='CoCo Chinese Chess 4 keypoints',
    paper_info=dict(
        author='theone',
        title='Chinese Chess Layout: ' +
        'A Dataset for Chinese Chess Layout',
        container='',
        year='2024',
        homepage='',
        version='1.0 (2024-12)',
        date_created='2024-12',
    ),
    # keypoint_info_array 2 dict
    keypoint_info=_keypoint_info,
    skeleton_info=_skeleton_info,
    joint_weights=_joint_weights,
    # 用于评估预测的关键点与真实值之间的误差容忍度
    sigmas=[0.025] * len(_bone_names),
)




@DATASETS.register_module(name='CChessDataset')
class CChessDataset(BaseCocoStyleDataset):
    METAINFO: dict = dataset_info
