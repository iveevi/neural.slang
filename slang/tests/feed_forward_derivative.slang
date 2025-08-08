import neural;

extern static const int In;
extern static const int Out;

typealias InputVec = InlineVector<float, In>;
typealias OutputVec = InlineVector<float, Out>;

// Single feed forward layer
struct FeedForwardGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<InputVec> dinput;
    
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> dparameters;
}

ParameterBlock<FeedForwardGlobals> globals;

[BackwardDifferentiable]
OutputVec feed_forward(StructuredBufferStorage<float> parameters, InputVec input)
{
    var network = FeedForward<float, InputVec.Dimension, OutputVec.Dimension, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    return network.eval<InputVec, OutputVec>(parameters, input);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    let parameters = StructuredBufferStorage<float>(globals.parameters);
    let dparameters = StructuredBufferStorage<float>(globals.dparameters);
    var dinput = diffPair(globals.input[tid]);
    var dff = DifferentialPtrPair<StructuredBufferStorage<float>>(parameters, dparameters);
    bwd_diff(feed_forward)(dff, dinput, OutputVec(1.0));
    globals.dinput[tid] = dinput.d;
}