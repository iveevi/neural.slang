import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// Mean squared error for vector of In elements
struct MSEGlobals
{
    RWStructuredBuffer<Vec> input;
    RWStructuredBuffer<Vec> target;
    RWStructuredBuffer<float> output;
    uint count;
}

ParameterBlock<MSEGlobals> globals;

static MeanSquaredError<float> mse = MeanSquaredError<float>();

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    var input = globals.input[tid];
    var target = globals.target[tid];
    var loss = mse.eval<In, Vec>(input, target);
    globals.output[tid] = loss;
}