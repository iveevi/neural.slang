import neural;

// Type aliases
typealias Vec4 = InlineVector<float, 4>;
typealias Vec8 = InlineVector<float, 8>;

// Network with 4 -> 8 -> 8 -> 4 layers
struct NetworkWithoutEncodingGlobals
{
    RWStructuredBuffer<Vec4> input;
    RWStructuredBuffer<Vec4> output;
    
    RWStructuredBuffer<float> layer1;
    RWStructuredBuffer<float> layer2;
    RWStructuredBuffer<float> layer3;
}

ParameterBlock<NetworkWithoutEncodingGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;

    let layer1 = StructuredBufferStorage<float>(globals.layer1);
    let layer2 = StructuredBufferStorage<float>(globals.layer2);
    let layer3 = StructuredBufferStorage<float>(globals.layer3);

    let ff1 = FeedForward<float, 4, 8, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let ff2 = FeedForward<float, 8, 8, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let ff3 = FeedForward<float, 8, 4, StructuredBufferStorage<float>, ReLU<float>>(0, {});

    let input = Vec4(globals.input[tid]);

    let x1 = ff1.eval<Vec4, Vec8>(layer1, input);
    let x2 = ff2.eval<Vec8, Vec8>(layer2, x1);
    let output = ff3.eval<Vec8, Vec4>(layer3, x2);

    globals.output[tid] = output;
}