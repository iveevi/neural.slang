import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// MAE derivative for vector of In elements
struct MAEDerivativeGlobals
{
    RWStructuredBuffer<Vec> predicted;
    RWStructuredBuffer<Vec> expected;
    RWStructuredBuffer<Vec> output;
}

ParameterBlock<MAEDerivativeGlobals> globals;

[BackwardDifferentiable]
float eval_mae(Vec predicted, no_diff Vec expected)
{
    static let mae_loss = MeanAbsoluteError<float>();
    return mae_loss.eval<In, Vec>(predicted, expected);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var predicted = globals.predicted[tid];
    var expected = globals.expected[tid];
    var dpredicted = diffPair(predicted);
    bwd_diff(eval_mae)(dpredicted, expected, 1.0);
    globals.output[tid] = dpredicted.d;
}
