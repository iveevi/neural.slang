import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// Mean absolute error for vector of In elements
struct MAEGlobals
{
    RWStructuredBuffer<Vec> input;
    RWStructuredBuffer<Vec> target;
    RWStructuredBuffer<float> output;
}

ParameterBlock<MAEGlobals> globals;

static MeanAbsoluteError<float> mae = MeanAbsoluteError<float>();

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var input = globals.input[tid];
    var target = globals.target[tid];
    var loss = mae.eval<In, Vec>(input, target);
    globals.output[tid] = loss;
}
