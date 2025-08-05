import neural;

// ReLU for scalars
struct ReLUGlobals
{
    RWStructuredBuffer<float> input;
    RWStructuredBuffer<float> output;
}

ParameterBlock<ReLUGlobals> relu_globals;

[shader("compute")]
void relu_main(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = ReLU<float>();
    var input = InlineVector<float, 1>(relu_globals.input[tid]);
    var output = activation.eval(input);
    relu_globals.output[tid] = output[0];
}
