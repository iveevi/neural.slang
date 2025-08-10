import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// ReLU for vector of In elements
struct ReLUGlobals
{
    RWStructuredBuffer<Vec> input;
    RWStructuredBuffer<Vec> output;
}

ParameterBlock<ReLUGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var input = globals.input[tid];
    var output = ReLU<float>().eval(input);
    globals.output[tid] = output;
}