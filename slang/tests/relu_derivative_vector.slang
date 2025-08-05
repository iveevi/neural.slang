import neural;

// ReLU derivative for vector of 2 elements
struct VectorReLUDerivativeGlobals
{
    RWStructuredBuffer<float2> input;
    RWStructuredBuffer<float2> output;
}

ParameterBlock<VectorReLUDerivativeGlobals> vector_relu_derivative_globals;

[BackwardDifferentiable]
InlineVector<float, 2> eval(no_diff ReLU<float> activation, InlineVector<float, 2> input)
{
    return activation.eval(input);
}

[shader("compute")]
void vector_relu_derivative_main(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = ReLU<float>();
    var input = InlineVector<float, 2>(vector_relu_derivative_globals.input[tid]);
    var dinput = diffPair(input, InlineVector<float, 2>(0.0));
    bwd_diff(eval)(activation, dinput, InlineVector<float, 2>(1.0));
    vector_relu_derivative_globals.output[tid] = float2(dinput.d[0], dinput.d[1]);
}