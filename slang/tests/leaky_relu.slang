import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// Leaky ReLU for vector of In elements
struct LeakyReLUGlobals
{
    RWStructuredBuffer<Vec> input;
    RWStructuredBuffer<Vec> output;
    RWStructuredBuffer<float> alpha;
}

ParameterBlock<LeakyReLUGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var input = globals.input[tid];
    var alpha = globals.alpha[0];
    var output = LeakyReLU<float>(alpha).eval(input);
    globals.output[tid] = output;
}
