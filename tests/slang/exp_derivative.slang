import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// Exp derivative for vector of In elements
struct ExpDerivativeGlobals
{
    RWStructuredBuffer<Vec> input;
    RWStructuredBuffer<Vec> output;
}

ParameterBlock<ExpDerivativeGlobals> globals;

[BackwardDifferentiable]
Vec eval(no_diff Exp<float> activation, Vec input)
{
    return activation.eval(input);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = Exp<float>();
    var input = Vec(globals.input[tid]);
    var dinput = diffPair(input, Vec(0.0));
    bwd_diff(eval)(activation, dinput, Vec(1.0));
    globals.output[tid] = dinput.d;
}
