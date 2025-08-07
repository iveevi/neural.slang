import neural;

// MSE derivative for vector of 16 elements
typealias Vec16 = InlineVector<float, 16>;

struct MSEDerivativeGlobals
{
    RWStructuredBuffer<Vec16> predicted;
    RWStructuredBuffer<Vec16> expected;
    RWStructuredBuffer<Vec16> output;
}

ParameterBlock<MSEDerivativeGlobals> globals;

[BackwardDifferentiable]
float eval_mse(Vec16 predicted, no_diff Vec16 expected)
{
    static let mse_loss = MeanSquaredError<float>();
    return mse_loss.eval<16, Vec16>(predicted, expected);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var predicted = globals.predicted[tid];
    var expected = globals.expected[tid];
    var dpredicted = diffPair(predicted, Vec16());
    bwd_diff(eval_mse)(dpredicted, expected, 1.0);
    globals.output[tid] = dpredicted.d;
}