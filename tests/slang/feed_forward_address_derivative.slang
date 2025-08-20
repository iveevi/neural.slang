import neural;

extern static const int In;
extern static const int Out;

typealias InputVec = InlineVector<float, In>;
typealias OutputVec = InlineVector<float, Out>;

// Single feed forward layer with address offset
struct FeedForwardGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<InputVec> dinput;
    
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> dparameters;

    uint address;
    uint count;
}

ParameterBlock<FeedForwardGlobals> globals;

[BackwardDifferentiable]
OutputVec feed_forward(StructuredBufferStorage<float> parameters, InputVec input)
{
    var network = FFLayer<float, In, Out, StructuredBufferStorage<float>, ReLU<float>>(globals.address, {});
    return network.eval<InputVec, OutputVec>(parameters, input);
}

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    if (tid >= globals.count) return;

    let parameters = StructuredBufferStorage<float>(globals.parameters);
    let dparameters = StructuredBufferStorage<float>(globals.dparameters);
    var dinput = diffPair(globals.input[tid]);
    var dff = DifferentialPtrPair<StructuredBufferStorage<float>>(parameters, dparameters);
    bwd_diff(feed_forward)(dff, dinput, OutputVec(1.0));
    globals.dinput[tid] = dinput.d;
}
