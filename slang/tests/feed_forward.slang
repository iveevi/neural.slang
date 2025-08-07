import neural;

typealias Vec4 = InlineVector<float, 4>;
typealias Vec8 = InlineVector<float, 8>;

// Single feed forward layer
struct FeedForwardGlobals
{
    RWStructuredBuffer<Vec4> input;
    RWStructuredBuffer<Vec8> output;
    
    RWStructuredBuffer<float> parameters;
}

ParameterBlock<FeedForwardGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    let parameters = StructuredBufferStorage<float>(globals.parameters);
    let network = FeedForward<float, 4, 8, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let input = Vec4(globals.input[tid]);
    let output = network.eval<Vec4, Vec8>(parameters, input);
    globals.output[tid] = output;
}