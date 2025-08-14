import neural;

extern static const int In;
extern static const int Hidden;
extern static const int Out;

typealias InputVec = InlineVector<float, In>;
typealias HiddenVec = InlineVector<float, Hidden>;
typealias OutputVec = InlineVector<float, Out>;

// Network with In -> Hidden -> Hidden -> Hidden -> Out layers
typealias Layer<int In, int Out> = FeedForward<float, In, Out, StructuredBufferStorage<float>, ReLU<float>>;

struct NetworkWithoutEncodingGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<OutputVec> output;
    
    RWStructuredBuffer<float> layer1;
    RWStructuredBuffer<float> layer2;
    RWStructuredBuffer<float> layer3;
    RWStructuredBuffer<float> layer4;
    uint count;
}

ParameterBlock<NetworkWithoutEncodingGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;


    let layer1 = StructuredBufferStorage<float>(globals.layer1);
    let layer2 = StructuredBufferStorage<float>(globals.layer2);
    let layer3 = StructuredBufferStorage<float>(globals.layer3);
    let layer4 = StructuredBufferStorage<float>(globals.layer4);

    let ff1 = Layer<In, Hidden>(0, {});
    let ff2 = Layer<Hidden, Hidden>(0, {});
    let ff3 = Layer<Hidden, Hidden>(0, {});
    let ff4 = Layer<Hidden, Out>(0, {});

    let input = InputVec(globals.input[tid]);

    let x1 = ff1.eval<InputVec, HiddenVec>(layer1, input);
    let x2 = ff2.eval<HiddenVec, HiddenVec>(layer2, x1);
    let x3 = ff3.eval<HiddenVec, HiddenVec>(layer3, x2);
    let output = ff4.eval<HiddenVec, OutputVec>(layer4, x3);

    globals.output[tid] = output;
}