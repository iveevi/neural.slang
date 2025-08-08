import neural;

typealias Vec128 = InlineVector<float, 128>;
typealias Vec64 = InlineVector<float, 64>;

// Single feed forward layer
struct FeedForwardGlobals
{
    RWStructuredBuffer<Vec128> input;
    RWStructuredBuffer<Vec128> dinput;
    
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> dparameters;
}

ParameterBlock<FeedForwardGlobals> globals;

[BackwardDifferentiable]
Vec64 feed_forward(StructuredBufferStorage<float> parameters, Vec128 input)
{
    var network = FeedForward<float, 128, 64, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    return network.eval<Vec128, Vec64>(parameters, input);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    let parameters = StructuredBufferStorage<float>(globals.parameters);
    let dparameters = StructuredBufferStorage<float>(globals.dparameters);
    var dinput = diffPair(globals.input[tid]);
    var dff = DifferentialPtrPair<StructuredBufferStorage<float>>(parameters, dparameters);
    bwd_diff(feed_forward)(dff, dinput, Vec64(1.0));
    globals.dinput[tid] = dinput.d;
}