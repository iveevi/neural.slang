import neural;

extern static const int In;
extern static const int Hidden;
extern static const int Out;

typealias InputVec = InlineVector<float, In>;
typealias HiddenVec = InlineVector<float, Hidden>;
typealias OutputVec = InlineVector<float, Out>;

// Network with In -> Hidden -> Hidden -> Hidden -> Out layers
typealias Layer<int In, int Out> = FFLayer<float, In, Out, StructuredBufferStorage<float>, ReLU<float>>;

struct NetworkWithoutEncodingGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<OutputVec> output;
    
    RWStructuredBuffer<float> parameters;

    uint layer1Address;
    uint layer2Address;
    uint layer3Address;
    uint layer4Address;
    uint count;
}

ParameterBlock<NetworkWithoutEncodingGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;


    let parameters = StructuredBufferStorage<float>(globals.parameters);

    let ff1 = Layer<In, Hidden>(globals.layer1Address, {});
    let ff2 = Layer<Hidden, Hidden>(globals.layer2Address, {});
    let ff3 = Layer<Hidden, Hidden>(globals.layer3Address, {});
    let ff4 = Layer<Hidden, Out>(globals.layer4Address, {});

    let input = InputVec(globals.input[tid]);

    let x1 = ff1.eval<InputVec, HiddenVec>(parameters, input);
    let x2 = ff2.eval<HiddenVec, HiddenVec>(parameters, x1);
    let x3 = ff3.eval<HiddenVec, HiddenVec>(parameters, x2);
    let output = ff4.eval<HiddenVec, OutputVec>(parameters, x3);

    globals.output[tid] = output;
}