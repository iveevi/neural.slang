import neural;

// Single feed forward layer
typealias Vec4 = InlineVector<float, 4>;
typealias Vec8 = InlineVector<float, 8>;

struct FeedForwardGlobals
{
    RWStructuredBuffer<Vec4> input;
    RWStructuredBuffer<Vec8> output;
    
    RWStructuredBuffer<float> parameters;
}

ParameterBlock<FeedForwardGlobals> feed_forward_globals;

[shader("compute")]
void feed_forward_main(uint3 thread_id : SV_DispatchThreadID)
{
    let parameters = StructuredBufferStorage<float>(feed_forward_globals.parameters);
    let tid = thread_id.x;
    let network = FeedForward<float, 4, 8, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let input = Vec4(feed_forward_globals.input[tid]);
    let output = network.eval<Vec4, Vec8>(parameters, input);
    feed_forward_globals.output[tid] = output;
}