codebase_config = dict(
    type='cchess_reg',  # 改为自定义类型
    task='Classification',
    model_type='end2end',
    model_class='CChessClassificationModel',  # 使用自定义模型类
    module=['cchess_reg.deploy'],  # 改为列表形式
)

backend_config = dict(type='onnxruntime')

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=None,
    optimize=True,
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'output': {
            0: 'batch',
            1: 'layout',
            2: 'cls'
        }
    })