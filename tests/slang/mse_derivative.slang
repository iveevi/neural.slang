import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// MSE derivative for vector of In elements
struct MSEDerivativeGlobals
{
    RWStructuredBuffer<Vec> predicted;
    RWStructuredBuffer<Vec> expected;
    RWStructuredBuffer<Vec> output;
    uint count;
}

ParameterBlock<MSEDerivativeGlobals> globals;

[BackwardDifferentiable]
float eval_mse(Vec predicted, no_diff Vec expected)
{
    static let mse_loss = MeanSquaredError<float>();
    return mse_loss.eval<In, Vec>(predicted, expected);
}

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    var predicted = globals.predicted[tid];
    var expected = globals.expected[tid];
    var dpredicted = diffPair(predicted);
    bwd_diff(eval_mse)(dpredicted, expected, 1.0);
    globals.output[tid] = dpredicted.d;
}