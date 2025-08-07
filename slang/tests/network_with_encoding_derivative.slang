import neural;

// Type aliases
typealias Vec3 = InlineVector<float, 3>;
typealias Vec24 = InlineVector<float, 24>;  // FrequencyEncoder<float, 3, 4> outputs 2*4*3=24 dims
typealias Vec8 = InlineVector<float, 8>;
typealias Vec4 = InlineVector<float, 4>;

// Network with frequency encoding derivative: 3 -> [FrequencyEncoder] -> 24 -> 8 -> 8 -> 4
struct NetworkWithEncodingDerivativeGlobals
{
    RWStructuredBuffer<Vec3> input;
    RWStructuredBuffer<Vec3> dinput;
    
    RWStructuredBuffer<float> layer1;
    RWStructuredBuffer<float> dlayer1;
    RWStructuredBuffer<float> layer2;
    RWStructuredBuffer<float> dlayer2;
    RWStructuredBuffer<float> layer3;
    RWStructuredBuffer<float> dlayer3;
}

ParameterBlock<NetworkWithEncodingDerivativeGlobals> globals;

[BackwardDifferentiable]
Vec4 network_with_encoding_forward(
    StructuredBufferStorage<float> layer1,
    StructuredBufferStorage<float> layer2, 
    StructuredBufferStorage<float> layer3,
    Vec3 input)
{
    // Create frequency encoder (3D input, 4 levels -> 24D output)
    let encoder = FrequencyEncoder<float, 3, 4>();
    
    // Create feed forward layers
    let ff1 = FeedForward<float, 24, 8, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let ff2 = FeedForward<float, 8, 8, StructuredBufferStorage<float>, ReLU<float>>(0, {});
    let ff3 = FeedForward<float, 8, 4, StructuredBufferStorage<float>, ReLU<float>>(0, {});

    // Apply frequency encoding
    let encoded = encoder.eval<Vec3, Vec24>(input);
    
    // Pass through neural network layers
    let x1 = ff1.eval<Vec24, Vec8>(layer1, encoded);
    let x2 = ff2.eval<Vec8, Vec8>(layer2, x1);
    let output = ff3.eval<Vec8, Vec4>(layer3, x2);

    return output;
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;

    let layer1 = StructuredBufferStorage<float>(globals.layer1);
    let layer2 = StructuredBufferStorage<float>(globals.layer2);
    let layer3 = StructuredBufferStorage<float>(globals.layer3);
    
    let dlayer1 = StructuredBufferStorage<float>(globals.dlayer1);
    let dlayer2 = StructuredBufferStorage<float>(globals.dlayer2);
    let dlayer3 = StructuredBufferStorage<float>(globals.dlayer3);

    var dinput = diffPair(globals.input[tid]);
    var dlayer1_pair = DifferentialPtrPair<StructuredBufferStorage<float>>(layer1, dlayer1);
    var dlayer2_pair = DifferentialPtrPair<StructuredBufferStorage<float>>(layer2, dlayer2);
    var dlayer3_pair = DifferentialPtrPair<StructuredBufferStorage<float>>(layer3, dlayer3);

    bwd_diff(network_with_encoding_forward)(dlayer1_pair, dlayer2_pair, dlayer3_pair, dinput, Vec4(1.0));
    
    globals.dinput[tid] = dinput.d;
}