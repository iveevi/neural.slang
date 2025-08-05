import pathlib
import slangpy as spy
import numpy as np
import argparse

np.set_printoptions(threshold=10000, linewidth=10000)

class EntryPoint:
    RELU = "relu"
    VECTOR_RELU = "vector_relu"
    VECTOR_RELU_DERIVATIVE = "vector_relu_derivative"
    MSE = "mse"
    FEED_FORWARD = "feed_forward"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--program",
    type=str,
    choices=[
        EntryPoint.RELU,
        EntryPoint.VECTOR_RELU,
        EntryPoint.VECTOR_RELU_DERIVATIVE,
        EntryPoint.MSE,
        EntryPoint.FEED_FORWARD,
    ],
    default=EntryPoint.RELU,
)
args = parser.parse_args()

program = args.program

device = spy.create_device(
    include_paths=[
        pathlib.Path(__file__).parent.absolute() / "slang",
    ],
)

program = device.load_program(
    "test.slang",
    entry_point_names=[args.program + "_main"],
)

kernel = device.create_compute_kernel(program)

def relu_program():
    data = 2 * np.random.rand(10).astype(np.float32) - 1

    input = device.create_buffer(
        size=data.nbytes,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
        data=data,
    )
    print("input:", input.to_numpy().view(np.float32))

    output = device.create_buffer(
        size=data.nbytes,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
    )

    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "relu_globals": {
                "input": input,
                "output": output,
            }
        },
    )

    output = output.to_numpy().view(np.float32)
    expected = np.where(data > 0, data, 0)
    error = np.abs(output - expected).sum()
    print("output:", output)
    print("expected:", expected)
    print("error:", error)

def vector_relu_program():
    data = 2 * np.random.rand(10, 2).astype(np.float32) - 1

    input = device.create_buffer(
        size=data.nbytes,
        struct_size=8,
        usage=spy.BufferUsage.shader_resource,
        data=data,
    )
    print("input:", input.to_numpy().view(np.float32).reshape(10, 2))

    output = device.create_buffer(
        size=data.nbytes,
        struct_size=8,
        usage=spy.BufferUsage.shader_resource,
    )

    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "vector_relu_globals": {
                "input": input,
                "output": output,
            }
        },
    )

    output = output.to_numpy().view(np.float32).reshape(10, 2)
    expected = np.where(data > 0, data, 0).reshape(10, 2)
    error = np.abs(output - expected).sum()
    print("output:", output)
    print("expected:", expected)
    print("error:", error)

def vector_relu_derivative_program():
    data = 2 * np.random.rand(10, 2).astype(np.float32) - 1

    input = device.create_buffer(
        size=data.nbytes,
        struct_size=8,
        usage=spy.BufferUsage.shader_resource,
        data=data,
    )
    print("input:", input.to_numpy().view(np.float32).reshape(10, 2))

    output = device.create_buffer(
        size=data.nbytes,
        struct_size=8,
        usage=spy.BufferUsage.shader_resource,
    )

    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "vector_relu_derivative_globals": {
                "input": input,
                "output": output,
            }
        },
    )

    output = output.to_numpy().view(np.float32).reshape(10, 2)
    expected = np.where(data > 0, 1, 0).reshape(10, 2)
    error = np.abs(output - expected).sum()
    print("output:", output)
    print("expected:", expected)
    print("error:", error)

def mse_program():
    input_data = 2 * np.random.rand(10, 16).astype(np.float32) - 1
    target_data = 2 * np.random.rand(10, 16).astype(np.float32) - 1

    input = device.create_buffer(
        size=input_data.nbytes,
        struct_size=16 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    print("input:", input.to_numpy().view(np.float32).reshape(10, 16))

    target = device.create_buffer(
        size=target_data.nbytes,
        struct_size=16 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=target_data,
    )
    print("target:", target.to_numpy().view(np.float32).reshape(10, 16))

    output = device.create_buffer(
        size=10 * 4,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
    )

    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "mse_globals": {
                "input": input,
                "target": target,
                "output": output,
            }
        },
    )

    expected = np.mean(np.square(input_data - target_data), axis=1)
    error = np.abs(output.to_numpy().view(np.float32) - expected).sum()
    print("output:", output.to_numpy().view(np.float32))
    print("output (numpy):", expected)
    print("error:", error)

def feed_forward_program():
    weights_data = 2 * np.random.rand(4, 8).astype(np.float32) - 1
    bias_data = 2 * np.random.rand(1, 8).astype(np.float32) - 1

    parameters_data = np.concatenate((weights_data, bias_data), axis=0)
    print("parameters:", parameters_data)

    input_data = 2 * np.random.rand(10, 4).astype(np.float32) - 1

    input = device.create_buffer(
        size=input_data.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    print("input:", input.to_numpy().view(np.float32).reshape(10, 4))

    output = device.create_buffer(
        size=10 * 8 * 4,
        struct_size=8 * 4,
        usage=spy.BufferUsage.shader_resource,
    )

    parameters = device.create_buffer(
        size=parameters_data.nbytes,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
        data=parameters_data,
    )

    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "feed_forward_globals": {
                "parameters": parameters,
                "input": input,
                "output": output,
            }
        },
    )

    output = output.to_numpy().view(np.float32).reshape(10, 8)
    expected = np.matmul(input_data, weights_data) + bias_data
    expected = np.where(expected > 0, expected, 0)
    error = np.abs(output - expected).sum()
    print("output:", output)
    print("expected:", expected)
    print("error:", error)

map = {
    EntryPoint.RELU: relu_program,
    EntryPoint.VECTOR_RELU: vector_relu_program,
    EntryPoint.VECTOR_RELU_DERIVATIVE: vector_relu_derivative_program,
    EntryPoint.MSE: mse_program,
    EntryPoint.FEED_FORWARD: feed_forward_program,
}

map[args.program]()