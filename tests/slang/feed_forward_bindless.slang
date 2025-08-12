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

    RWStructuredBuffer<float>.Handle weights;
    RWStructuredBuffer<float>.Handle biases;
}

ParameterBlock<FeedForwardGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    let weights = BindlessBufferStorage<float>(globals.weights, 0u);
    let biases = BindlessBufferStorage<float>(globals.biases, 0u);
    let network = BindlessFeedForward<float, In, Out, BindlessBufferStorage<float>, ReLU<float>>(weights, biases, {});
    let input = InputVec(globals.input[tid]);
    let output = network.eval<InputVec, OutputVec>(input);
    globals.output[tid] = output;
}