import neural;

extern static const int In;
extern static const int Hidden;
extern static const int Out;

typealias InputVec = InlineVector<float, In>;
typealias HiddenVec = InlineVector<float, Hidden>;
typealias OutputVec = InlineVector<float, Out>;

// Network with In -> Hidden -> Hidden -> Hidden -> Out layers - derivative version
typealias Layer<int In, int Out> = FeedForward<float, In, Out, StructuredBufferStorage<float>, ReLU<float>>;

struct NetworkWithoutEncodingDerivativeGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<InputVec> dinput;
    
    RWStructuredBuffer<float> layer1;
    RWStructuredBuffer<float> dlayer1;
    RWStructuredBuffer<float> layer2;
    RWStructuredBuffer<float> dlayer2;
    RWStructuredBuffer<float> layer3;
    RWStructuredBuffer<float> dlayer3;
    RWStructuredBuffer<float> layer4;
    RWStructuredBuffer<float> dlayer4;
}

ParameterBlock<NetworkWithoutEncodingDerivativeGlobals> globals;

[BackwardDifferentiable]
OutputVec network_forward(
    StructuredBufferStorage<float> layer1,
    StructuredBufferStorage<float> layer2, 
    StructuredBufferStorage<float> layer3,
    StructuredBufferStorage<float> layer4,
    InputVec input)
{
    let ff1 = Layer<In, Hidden>(0, {});
    let ff2 = Layer<Hidden, Hidden>(0, {});
    let ff3 = Layer<Hidden, Hidden>(0, {});
    let ff4 = Layer<Hidden, Out>(0, {});

    let x1 = ff1.eval<InputVec, HiddenVec>(layer1, input);
    let x2 = ff2.eval<HiddenVec, HiddenVec>(layer2, x1);
    let x3 = ff3.eval<HiddenVec, HiddenVec>(layer3, x2);
    let output = ff4.eval<HiddenVec, OutputVec>(layer4, x3);

    return output;
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;

    let layer1 = StructuredBufferStorage<float>(globals.layer1);
    let layer2 = StructuredBufferStorage<float>(globals.layer2);
    let layer3 = StructuredBufferStorage<float>(globals.layer3);
    let layer4 = StructuredBufferStorage<float>(globals.layer4);
    
    let dlayer1 = StructuredBufferStorage<float>(globals.dlayer1);
    let dlayer2 = StructuredBufferStorage<float>(globals.dlayer2);
    let dlayer3 = StructuredBufferStorage<float>(globals.dlayer3);
    let dlayer4 = StructuredBufferStorage<float>(globals.dlayer4);

    var dinput = diffPair(globals.input[tid]);
    var dlayer1_pair = DifferentialPtrPair<StructuredBufferStorage<float>>(layer1, dlayer1);
    var dlayer2_pair = DifferentialPtrPair<StructuredBufferStorage<float>>(layer2, dlayer2);
    var dlayer3_pair = DifferentialPtrPair<StructuredBufferStorage<float>>(layer3, dlayer3);
    var dlayer4_pair = DifferentialPtrPair<StructuredBufferStorage<float>>(layer4, dlayer4);

    bwd_diff(network_forward)(dlayer1_pair, dlayer2_pair, dlayer3_pair, dlayer4_pair, dinput, OutputVec(1.0));
    
    globals.dinput[tid] = dinput.d;
}