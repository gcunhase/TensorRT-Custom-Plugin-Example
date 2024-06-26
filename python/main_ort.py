import numpy as np
import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs

# import common_runtime


def load_dummy_data(input_shape, data_size=5):
    for _ in range(data_size):
        if isinstance(input_shape, list):
            yield [np.random.rand(*inp).astype(np.float32) for inp in input_shape]
        else:
            yield np.random.rand(*input_shape).astype(np.float32)


def main():

    onnx_file_path = "../data/identity_neural_network.onnx"
    plugin_lib_file_path = "../build/src/plugins/IdentityConvIPluginV2IOExt/libidentity_conv_iplugin_v2_io_ext.so"

    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 1

    # Create TensorRT provider options
    assert "TensorrtExecutionProvider" in ort.get_available_providers(), "TensorrtExecutionProvider not available!"

    EP = [
        ('TensorrtExecutionProvider', {
            "device_id": 0,
            "trt_extra_plugin_lib_paths": plugin_lib_file_path,
        }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
    ]

    # Dummy example.
    # Generate random data and get input / output shapes
    graph = gs.import_onnx(onnx.load(onnx_file_path))
    if len(graph.inputs) == 1:
        input_shape = tuple(graph.inputs[0].shape)
    else:
        input_shape = [inp.shape for inp in graph.inputs]
    if len(graph.outputs) == 1:
        output_shape = tuple(graph.outputs[0].shape)
    else:
        output_shape = [out.shape for out in graph.outputs]
    data = load_dummy_data(input_shape=input_shape)

    # Initiate ORT session
    ort_session = ort.InferenceSession(onnx_file_path, sess_options=session_opts, providers=EP)
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]

    # Print input tensor information
    print("Input Tensors:")
    print(" - Names: {}".format(input_names))
    print(" - Shapes: {}".format(input_shape))

    # Execute ORT
    inputs = []
    outputs = []
    print("Inference: ")
    for i, images in enumerate(data):
        print(f"  Batch {i + 1}")
        inputs.append([images])
        inp = dict(zip(input_names, images if len(input_names) > 1 else [images]))
        output = ort_session.run([], inp)
        outputs.append(output)

    # Print output tensor data.
    print("Output Tensors:")
    print(" - Names: {}".format(output_names))
    print(" - Shapes: {}".format(output_shape))

    # In our case, the input and output tensor data should be exactly the same.
    try:
        for inp, out in zip(inputs, outputs):
            np.testing.assert_equal(inp, out)
        print("PASS: All inputs / outputs match!")
    except AssertionError:
        print("FAIL: At least some inputs / outputs don't match!")


if __name__ == "__main__":

    main()
