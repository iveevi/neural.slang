import neural;

typealias Vec16 = InlineVector<float, 16>;

// ReLU for vector of 16 elements
struct VectorReLUGlobals
{
    RWStructuredBuffer<Vec16> input;
    RWStructuredBuffer<Vec16> output;
}

ParameterBlock<VectorReLUGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = ReLU<float>();
    var input = globals.input[tid];
    var output = activation.eval(input);
    globals.output[tid] = Vec16(output);
}