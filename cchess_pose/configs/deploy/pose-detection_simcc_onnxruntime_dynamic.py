backend_config = dict(type='onnxruntime')

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='cchess_pose4.onnx',
    input_names=['input'],
    input_shape=[256, 256],
    optimize=True,
    output_names=['simcc_x', 'simcc_y'],
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'simcc_x': {
            0: 'batch'
        },
        'simcc_y': {
            0: 'batch'
        }
    }
)

codebase_config = dict(
    type='mmpose', task='PoseDetection',
    export_postprocess=False  # do not export get_simcc_maximum
)
