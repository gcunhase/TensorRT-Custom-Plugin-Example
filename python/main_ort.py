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
    plugin_lib_file_path = "../build/src/libidentity_conv.so"

    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 1

    # Create TensorRT provider options
    assert "TensorrtExecutionProvider" in ort.get_available_providers(), "TensorrtExecutionProvider not available!"
    tensorrt_options = {
        "device_id": 0,
        "trt_extra_plugin_lib_paths": plugin_lib_file_path,
    }

    EP = [('TensorrtExecutionProvider', tensorrt_options)]

    # import ctypes
    # ctypes.CDLL(self.custom_ops, winmode=0)
    # common_runtime.load_plugin_lib(plugin_lib_file_path)

    # Dummy example.
    # Generate random data
    graph = gs.import_onnx(onnx.load(onnx_file_path))
    if len(graph.inputs) == 1:
        input_shape = tuple(graph.inputs[0].shape)
    else:
        input_shape = [inp.shape for inp in graph.inputs]
    data = load_dummy_data(input_shape=input_shape)

    # Initiate ORT session
    ort_session = ort.InferenceSession(onnx_file_path, sess_options=session_opts, providers=EP)
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [inp.name for inp in ort_session.get_outputs()]

    # Print input tensor information
    print("Input Tensor:")
    print(" - Names: {}".format(input_names))
    print(" - Shapes: {}".format(input_shape))

    # Execute ORT
    outputs = []
    for i, images in enumerate(data):
        print(f"Batch {i + 1}")
        inp = dict(zip(input_names, images if len(input_names) > 1 else [images]))
        output = ort_session.run([], inp)
        outputs.append(output)

    # Print output tensor data.
    print("Outputs Tensor:")
    print(" - Names: {}".format(output_names))

    # # In our case, the input and output tensor data should be exactly the same.
    # for input_host_device_buffer, output_host_device_buffer in zip(
    #         inputs, outputs):
    #     np.testing.assert_equal(input_host_device_buffer.host,
    #                             output_host_device_buffer.host)


if __name__ == "__main__":

    main()
