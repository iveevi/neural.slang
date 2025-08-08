import neural;

typealias Vec128 = InlineVector<float, 128>;
typealias Vec64 = InlineVector<float, 64>;

// Single feed forward layer
struct FeedForwardGlobals
{
    RWStructuredBuffer<Vec128> input;
    RWStructuredBuffer<Vec64> output;
    
    RWStructuredBuffer<float> parameters;
}

ParameterBlock<FeedForwardGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    let parameters = StructuredBufferStorage<float>(globals.parameters);
    let network = FeedForward<float, 128, 64, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let input = Vec128(globals.input[tid]);
    let output = network.eval<Vec128, Vec64>(parameters, input);
    globals.output[tid] = output;
}