import neural;

// Mean squared error for vector of 16 elements
typealias Vec16 = InlineVector<float, 16>;

struct MSEGlobals
{
    RWStructuredBuffer<Vec16> input;
    RWStructuredBuffer<Vec16> target;
    RWStructuredBuffer<float> output;
}

ParameterBlock<MSEGlobals> mse_globals;

static MeanSquaredError mse = MeanSquaredError();

[shader("compute")]
void mse_main(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var input = mse_globals.input[tid];
    var target = mse_globals.target[tid];
    var loss = mse.eval<float, 16, Vec16>(input, target);
    mse_globals.output[tid] = loss;
}