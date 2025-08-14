import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// Exp for vector of In elements
struct ExpGlobals
{
    RWStructuredBuffer<Vec> input;
    RWStructuredBuffer<Vec> output;
}

ParameterBlock<ExpGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var input = globals.input[tid];
    var output = Exp<float>().eval(input);
    globals.output[tid] = output;
}
