import neural;

typealias Vec4 = InlineVector<float, 4>;
typealias Vec8 = InlineVector<float, 8>;

// Single feed forward layer
struct FeedForwardGlobals
{
    RWStructuredBuffer<Vec4> input;
    RWStructuredBuffer<Vec4> dinput;
    
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> dparameters;
}

ParameterBlock<FeedForwardGlobals> globals;

[BackwardDifferentiable]
Vec8 feed_forward(StructuredBufferStorage<float> parameters, Vec4 input)
{
    var network = FeedForward<float, 4, 8, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    return network.eval<Vec4, Vec8>(parameters, input);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    let parameters = StructuredBufferStorage<float>(globals.parameters);
    let dparameters = StructuredBufferStorage<float>(globals.dparameters);
    var dinput = diffPair(globals.input[tid]);
    var dff = DifferentialPtrPair<StructuredBufferStorage<float>>(parameters, dparameters);
    bwd_diff(feed_forward)(dff, dinput, Vec8(1.0));
    globals.dinput[tid] = dinput.d;
}