import neural;

// ReLU for vector of 2 elements
struct VectorReLUGlobals
{
    RWStructuredBuffer<float2> input;
    RWStructuredBuffer<float2> output;
}

ParameterBlock<VectorReLUGlobals> vector_relu_globals;

[shader("compute")]
void vector_relu_main(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = ReLU<float>();
    var input = InlineVector<float, 2>(vector_relu_globals.input[tid]);
    var output = activation.eval(input);
    vector_relu_globals.output[tid] = float2(output[0], output[1]);
}