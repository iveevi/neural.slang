import neural;

// ReLU for scalars
struct ReLUGlobals
{
    RWStructuredBuffer<float> input;
    RWStructuredBuffer<float> output;
}

ParameterBlock<ReLUGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = ReLU<float>();
    var input = InlineVector<float, 1>(globals.input[tid]);
    var output = activation.eval(input);
    globals.output[tid] = output[0];
}
