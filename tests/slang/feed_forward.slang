import neural;

extern static const int In;
extern static const int Out;

typealias InputVec = InlineVector<float, In>;
typealias OutputVec = InlineVector<float, Out>;

// Single feed forward layer
struct FeedForwardGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<OutputVec> output;
    
    RWStructuredBuffer<float> parameters;
    uint count;
}

ParameterBlock<FeedForwardGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    if (tid >= globals.count) return;

    let storage = StructuredBufferStorage<float>(globals.parameters);
    let network = FFLayer<float, In, Out, StructuredBufferStorage<float>, ReLU<float>>(0u, {});
    let input = InputVec(globals.input[tid]);
    let output = network.eval<InputVec, OutputVec>(storage, input);
    globals.output[tid] = output;
}