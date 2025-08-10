import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// Mean squared error for vector of In elements
struct MSEGlobals
{
    RWStructuredBuffer<Vec> input;
    RWStructuredBuffer<Vec> target;
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
    var loss = mse.eval<In, Vec>(input, target);
    globals.output[tid] = loss;
}