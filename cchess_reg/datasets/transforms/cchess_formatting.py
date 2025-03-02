from mmpretrain.datasets.transforms.formatting import PackInputs
from cchess_reg.structures import CChessDataSample
from mmpretrain.registry import TRANSFORMS

@TRANSFORMS.register_module()
class CChessPackInputs(PackInputs):
    """CChess 版 PackInputs 类
    
    继承自 mmpretrain 的 PackInputs 类，用于将预处理后的数据打包成模型所需的格式。
    主要功能是将图像数据和标签数据打包到 CChessDataSample 结构中，以便模型训练和推理。
    
    属性:
        input_key (str): 输入数据的键名，默认为 'img'
        algorithm_keys (list): 算法相关的键列表
        meta_keys (list): 元信息相关的键列表
    """
    
    def transform(self, results: dict) -> dict:
        """将输入数据打包成模型所需的格式。
        
        Args:
            results (dict): 包含预处理后数据的字典
            
        Returns:
            dict: 打包后的数据，包含 'inputs' 和 'data_samples' 两个键
                - inputs: 模型的输入数据，通常是图像张量
                - data_samples: CChessDataSample 实例，包含标签、掩码等信息
        """

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)

        # 创建中国象棋数据样本实例
        data_sample = CChessDataSample()

        # 设置基础标签信息
        if 'gt_label' in results:
            data_sample.set_gt_label(results['gt_label'])  # 设置真实标签
        if 'gt_score' in results:
            data_sample.set_gt_score(results['gt_score'])  # 设置真实分数
        if 'mask' in results:
            data_sample.set_mask(results['mask'])  # 设置掩码信息

        # 设置自定义算法所需的额外信息
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # 设置元信息，如图像路径、尺寸等
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')

        packed_results['data_samples'] = data_sample
        return packed_results

