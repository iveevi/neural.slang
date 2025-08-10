import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// Leaky ReLU derivative for vector of In elements
struct LeakyReLUDerivativeGlobals
{
    RWStructuredBuffer<Vec> input;
    RWStructuredBuffer<Vec> output;
    RWStructuredBuffer<float> alpha;
}

ParameterBlock<LeakyReLUDerivativeGlobals> globals;

[BackwardDifferentiable]
Vec eval(no_diff LeakyReLU<float> activation, Vec input)
{
    return activation.eval(input);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var alpha = globals.alpha[0];
    var activation = LeakyReLU<float>(alpha);
    var input = Vec(globals.input[tid]);
    var dinput = diffPair(input, Vec(0.0));
    bwd_diff(eval)(activation, dinput, Vec(1.0));
    globals.output[tid] = dinput.d;
}
