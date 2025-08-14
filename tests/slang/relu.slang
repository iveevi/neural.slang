import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// ReLU for vector of In elements
struct ReLUGlobals
{
    RWStructuredBuffer<Vec> input;
    RWStructuredBuffer<Vec> output;
    uint count;
}

ParameterBlock<ReLUGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    var input = globals.input[tid];
    var output = ReLU<float>().eval(input);
    globals.output[tid] = output;
}