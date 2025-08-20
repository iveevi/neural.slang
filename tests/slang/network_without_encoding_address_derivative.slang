import neural;

extern static const int In;
extern static const int Hidden;
extern static const int Out;

typealias InputVec = InlineVector<float, In>;
typealias HiddenVec = InlineVector<float, Hidden>;
typealias OutputVec = InlineVector<float, Out>;

// Network with In -> Hidden -> Hidden -> Hidden -> Out layers - derivative version with addresses
struct NetworkWithoutEncodingAddressDerivativeGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<InputVec> dinput;
    
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> dparameters;

    uint layer1Address;
    uint layer2Address;
    uint layer3Address;
    uint layer4Address;
    uint count;
}

ParameterBlock<NetworkWithoutEncodingAddressDerivativeGlobals> globals;

[BackwardDifferentiable]
OutputVec network_forward(StructuredBufferStorage<float> parameters, InputVec input)
{
    let ff1 = FFLayer<float, In, Hidden, StructuredBufferStorage<float>, ReLU<float>>(globals.layer1Address, {});
    let ff2 = FFLayer<float, Hidden, Hidden, StructuredBufferStorage<float>, ReLU<float>>(globals.layer2Address, {});
    let ff3 = FFLayer<float, Hidden, Hidden, StructuredBufferStorage<float>, ReLU<float>>(globals.layer3Address, {});
    let ff4 = FFLayer<float, Hidden, Out, StructuredBufferStorage<float>, ReLU<float>>(globals.layer4Address, {});

    let x1 = ff1.eval<InputVec, HiddenVec>(parameters, input);
    let x2 = ff2.eval<HiddenVec, HiddenVec>(parameters, x1);
    let x3 = ff3.eval<HiddenVec, HiddenVec>(parameters, x2);
    let output = ff4.eval<HiddenVec, OutputVec>(parameters, x3);

    return output;
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
    var dparameters_pair = DifferentialPtrPair<StructuredBufferStorage<float>>(parameters, dparameters);
    bwd_diff(network_forward)(dparameters_pair, dinput, OutputVec(1.0));
    globals.dinput[tid] = dinput.d;
}
