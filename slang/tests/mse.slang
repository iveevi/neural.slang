import neural;

// Mean squared error for vector of 16 elements
typealias Vec16 = InlineVector<float, 16>;

struct MSEGlobals
{
    RWStructuredBuffer<Vec16> input;
    RWStructuredBuffer<Vec16> target;
    RWStructuredBuffer<float> output;
}

ParameterBlock<MSEGlobals> globals;

static MeanSquaredError<float> mse = MeanSquaredError<float>();

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var input = globals.input[tid];
    var target = globals.target[tid];
    var loss = mse.eval<16, Vec16>(input, target);
    globals.output[tid] = loss;
}