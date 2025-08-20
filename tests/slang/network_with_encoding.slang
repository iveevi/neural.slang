import neural;

extern static const int In;
extern static const int Hidden;
extern static const int Levels;
extern static const int Out;

static const int Encoded = 2 * Levels * In;

typealias InputVec = InlineVector<float, In>;
typealias OutputVec = InlineVector<float, Out>;
typealias HiddenVec = InlineVector<float, Hidden>;
typealias EncodedVec = InlineVector<float, Encoded>;

// Network with frequency encoding: In -> [FrequencyEncoder] -> Encoded -> Hidden -> Hidden -> Hidden -> Out
typealias Layer<int In, int Out> = FFLayer<float, In, Out, StructuredBufferStorage<float>, ReLU<float>>;

struct NetworkWithEncodingGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<OutputVec> output;
    
    RWStructuredBuffer<float> layer1;  // Encoded -> Hidden
    RWStructuredBuffer<float> layer2;  // Hidden -> Hidden  
    RWStructuredBuffer<float> layer3;  // Hidden -> Hidden
    RWStructuredBuffer<float> layer4;  // Hidden -> Out
    uint count;
}

ParameterBlock<NetworkWithEncodingGlobals> globals;

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

    let encoder = FrequencyEncoder<float, In, Levels>();
    
    let ff1 = FFLayer<float, Encoded, Hidden, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let ff2 = Layer<Hidden, Hidden>(0, {});
    let ff3 = Layer<Hidden, Hidden>(0, {});
    let ff4 = Layer<Hidden, Out>(0, {});

    let input = InputVec(globals.input[tid]);

    let encoded = encoder.eval<InputVec, EncodedVec>(input);
    
    let x1 = ff1.eval<EncodedVec, HiddenVec>(layer1, encoded);
    let x2 = ff2.eval<HiddenVec, HiddenVec>(layer2, x1);
    let x3 = ff3.eval<HiddenVec, HiddenVec>(layer3, x2);
    let output = ff4.eval<HiddenVec, OutputVec>(layer4, x3);

    globals.output[tid] = output;
}