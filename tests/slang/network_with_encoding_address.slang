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

// Network with frequency encoding and addresses: In -> [FrequencyEncoder] -> Encoded -> Hidden -> Hidden -> Hidden -> Out
struct NetworkWithEncodingAddressGlobals
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

ParameterBlock<NetworkWithEncodingAddressGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;


    let parameters = StructuredBufferStorage<float>(globals.parameters);

    let encoder = FrequencyEncoder<float, In, Levels>();
    
    let ff1 = FFLayer<float, Encoded, Hidden, StructuredBufferStorage<float>, ReLU<float>>(globals.layer1Address, {});
    let ff2 = FFLayer<float, Hidden, Hidden, StructuredBufferStorage<float>, ReLU<float>>(globals.layer2Address, {});
    let ff3 = FFLayer<float, Hidden, Hidden, StructuredBufferStorage<float>, ReLU<float>>(globals.layer3Address, {});
    let ff4 = FFLayer<float, Hidden, Out, StructuredBufferStorage<float>, ReLU<float>>(globals.layer4Address, {});

    let input = InputVec(globals.input[tid]);

    let encoded = encoder.eval<InputVec, EncodedVec>(input);
    
    let x1 = ff1.eval<EncodedVec, HiddenVec>(parameters, encoded);
    let x2 = ff2.eval<HiddenVec, HiddenVec>(parameters, x1);
    let x3 = ff3.eval<HiddenVec, HiddenVec>(parameters, x2);
    let output = ff4.eval<HiddenVec, OutputVec>(parameters, x3);

    globals.output[tid] = output;
}
