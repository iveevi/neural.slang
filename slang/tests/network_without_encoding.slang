import neural;

// Type aliases
typealias Vec4 = InlineVector<float, 4>;
typealias Vec32 = InlineVector<float, 32>;

// Network with 4 -> 32 -> 32 -> 32 -> 4 layers
struct NetworkWithoutEncodingGlobals
{
    RWStructuredBuffer<Vec4> input;
    RWStructuredBuffer<Vec4> output;
    
    RWStructuredBuffer<float> layer1;
    RWStructuredBuffer<float> layer2;
    RWStructuredBuffer<float> layer3;
    RWStructuredBuffer<float> layer4;
}

ParameterBlock<NetworkWithoutEncodingGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;

    let layer1 = StructuredBufferStorage<float>(globals.layer1);
    let layer2 = StructuredBufferStorage<float>(globals.layer2);
    let layer3 = StructuredBufferStorage<float>(globals.layer3);
    let layer4 = StructuredBufferStorage<float>(globals.layer4);

    let ff1 = FeedForward<float, 4, 32, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let ff2 = FeedForward<float, 32, 32, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let ff3 = FeedForward<float, 32, 32, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let ff4 = FeedForward<float, 32, 4, StructuredBufferStorage<float>, ReLU<float>>(0, {});

    let input = Vec4(globals.input[tid]);

    let x1 = ff1.eval<Vec4, Vec32>(layer1, input);
    let x2 = ff2.eval<Vec32, Vec32>(layer2, x1);
    let x3 = ff3.eval<Vec32, Vec32>(layer3, x2);
    let output = ff4.eval<Vec32, Vec4>(layer4, x3);

    globals.output[tid] = output;
}