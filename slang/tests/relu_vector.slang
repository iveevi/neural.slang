import neural;

// ReLU for vector of 2 elements
struct VectorReLUGlobals
{
    RWStructuredBuffer<float2> input;
    RWStructuredBuffer<float2> output;
}

ParameterBlock<VectorReLUGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = ReLU<float>();
    var input = InlineVector<float, 2>(globals.input[tid]);
    var output = activation.eval(input);
    globals.output[tid] = float2(output[0], output[1]);
}