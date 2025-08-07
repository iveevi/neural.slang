import neural;

typealias Vec2 = InlineVector<float, 2>;

// ReLU derivative for vector of 2 elements
struct VectorReLUDerivativeGlobals
{
    RWStructuredBuffer<Vec2> input;
    RWStructuredBuffer<Vec2> output;
}

ParameterBlock<VectorReLUDerivativeGlobals> globals;

[BackwardDifferentiable]
Vec2 eval(no_diff ReLU<float> activation, Vec2 input)
{
    return activation.eval(input);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = ReLU<float>();
    var input = Vec2(globals.input[tid]);
    var dinput = diffPair(input, Vec2(0.0));
    bwd_diff(eval)(activation, dinput, Vec2(1.0));
    globals.output[tid] = dinput.d;
}