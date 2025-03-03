
codebase_config = dict(type='mmpretrain', task='Classification')
backend_config = dict(type='onnxruntime')

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='cchess_reg.onnx',
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
