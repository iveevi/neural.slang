import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// Sine for vector of In elements
struct SineGlobals
{
    RWStructuredBuffer<Vec> input;
    RWStructuredBuffer<Vec> output;
}

ParameterBlock<SineGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var input = globals.input[tid];
    var output = Sine<float>().eval(input);
    globals.output[tid] = output;
}
