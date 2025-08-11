import neural;

extern static const int In;
extern static const int Out;
extern static const int Hidden;
extern static const int Levels; // 0 means no (frequency) encoding

typealias InputVec = InlineVector<float, In>;
typealias OutputVec = InlineVector<float, Out>;
typealias HiddenVec = InlineVector<float, Hidden>;
typealias EncodedVec = InlineVector<float, 2 * In * Levels>;

typealias StorageType = StructuredBufferStorage<float>;
typealias StorageDiffType = DifferentialPtrPair<StorageType>;
typealias ActivationType = ReLU<float>;
typealias OptimizerType = Adam<float>;
// typealias OptimizerType = SGD<float>;
typealias Layer<int In, int Out> = FeedForward<float, In, Out, StorageType, ActivationType>;
typealias LastLayer<int In, int Out> = FeedForward<float, In, Out, StorageType, Identity<float>>;

// Neural network parameters
struct NetworkParameters
{
    RWStructuredBuffer<float> layer1;  // Encoded/In -> Hidden
    RWStructuredBuffer<float> layer2;  // Hidden -> Hidden  
    RWStructuredBuffer<float> layer3;  // Hidden -> Hidden
    RWStructuredBuffer<float> layer4;  // Hidden -> Out
}

ParameterBlock<NetworkParameters> parameters;

[BackwardDifferentiable]
OutputVec eval(
	StorageType layer1,
	StorageType layer2,
	StorageType layer3,
	StorageType layer4,
	no_diff InputVec input)
{
    let ff2 = Layer<Hidden, Hidden>(0, {});
    let ff3 = Layer<Hidden, Hidden>(0, {});
    let ff4 = LastLayer<Hidden, Out>(0, {});

    if (Levels == 0)
    {
        let ff1 = Layer<In, Hidden>(0, {});
        let x1 = ff1.eval<InputVec, HiddenVec>(layer1, input);
        let x2 = ff2.eval<HiddenVec, HiddenVec>(layer2, x1);
        let x3 = ff3.eval<HiddenVec, HiddenVec>(layer3, x2);
        let output = ff4.eval<HiddenVec, OutputVec>(layer4, x3);
        return output;
    }
    else
    {
        let encoder = FrequencyEncoder<float, In, Levels>();
        let encoded = encoder.eval<InputVec, EncodedVec>(input);

        let ff1 = Layer<2 * In * Levels, Hidden>(0, {});
        let x1 = ff1.eval<EncodedVec, HiddenVec>(layer1, encoded);
        let x2 = ff2.eval<HiddenVec, HiddenVec>(layer2, x1);
        let x3 = ff3.eval<HiddenVec, HiddenVec>(layer3, x2);
        let output = ff4.eval<HiddenVec, OutputVec>(layer4, x3);
        return output;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void forward(
    RWStructuredBuffer<InputVec> inputBuffer,
    RWStructuredBuffer<OutputVec> outputBuffer,
    uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;

    let layer1 = StructuredBufferStorage<float>(parameters.layer1);
    let layer2 = StructuredBufferStorage<float>(parameters.layer2);
    let layer3 = StructuredBufferStorage<float>(parameters.layer3);
    let layer4 = StructuredBufferStorage<float>(parameters.layer4);

    let input = InputVec(inputBuffer[tid]);
    let output = eval(layer1, layer2, layer3, layer4, input);
    outputBuffer[tid] = output;
}

// Neural network gradients
struct NetworkGradients
{
    RWStructuredBuffer<float> layer1;
    RWStructuredBuffer<float> layer2;
    RWStructuredBuffer<float> layer3;
    RWStructuredBuffer<float> layer4;
}

ParameterBlock<NetworkGradients> gradients;

[BackwardDifferentiable]
float loss(
	StorageType layer1,
	StorageType layer2,
	StorageType layer3,
	StorageType layer4,
	no_diff InputVec input,
	no_diff OutputVec expected)
{
    let mse = MeanSquaredError<float>();
    let output = eval(layer1, layer2, layer3, layer4, input);
    return mse.eval(output, expected);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void backward(
	RWStructuredBuffer<InputVec> inputBuffer,
    RWStructuredBuffer<OutputVec> expectedBuffer,
	uint3 thread_id : SV_DispatchThreadID,
    uniform float boost)
{
    int tid = thread_id.x;

    let layer1 = StructuredBufferStorage<float>(parameters.layer1);
    let layer2 = StructuredBufferStorage<float>(parameters.layer2);
    let layer3 = StructuredBufferStorage<float>(parameters.layer3);
    let layer4 = StructuredBufferStorage<float>(parameters.layer4);

    let dlayer1 = StructuredBufferStorage<float>(gradients.layer1);
    let dlayer2 = StructuredBufferStorage<float>(gradients.layer2);
    let dlayer3 = StructuredBufferStorage<float>(gradients.layer3);
    let dlayer4 = StructuredBufferStorage<float>(gradients.layer4);

    var layer1Diff = StorageDiffType(layer1, dlayer1);
    var layer2Diff = StorageDiffType(layer2, dlayer2);
    var layer3Diff = StorageDiffType(layer3, dlayer3);
    var layer4Diff = StorageDiffType(layer4, dlayer4);

    bwd_diff(loss)(
        layer1Diff,
        layer2Diff,
        layer3Diff,
        layer4Diff,
        inputBuffer[tid],
        expectedBuffer[tid],
        boost
    );
}

// Optimization
void update(
    RWStructuredBuffer<OptimizerType.State> states,
    RWStructuredBuffer<float> parameters,
    RWStructuredBuffer<float> gradients,
    uint tid,
    uint stride,
    uint count)
{
    let optimizer = OptimizerType();

    for (uint i = tid; i < count; i += stride) {
        optimizer.step(states[i], parameters[i], gradients[i]);
        gradients[i] = 0.0;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void optimize(
    RWStructuredBuffer<float> layer1Params,
    RWStructuredBuffer<float> layer1Grads,
    RWStructuredBuffer<OptimizerType.State> layer1States,
    RWStructuredBuffer<float> layer2Params,
    RWStructuredBuffer<float> layer2Grads,
    RWStructuredBuffer<OptimizerType.State> layer2States,
    RWStructuredBuffer<float> layer3Params,
    RWStructuredBuffer<float> layer3Grads,
    RWStructuredBuffer<OptimizerType.State> layer3States,
    RWStructuredBuffer<float> layer4Params,
    RWStructuredBuffer<float> layer4Grads,
    RWStructuredBuffer<OptimizerType.State> layer4States,
    uint3 thread_id : SV_DispatchThreadID,
    uniform uint dispatchSize,
    uniform uint layer1Count,
    uniform uint layer2Count,
    uniform uint layer3Count,
    uniform uint layer4Count)
{
    uint tid = thread_id.x;
    uint stride = dispatchSize;

    update(layer1States, layer1Params, layer1Grads, tid, stride, layer1Count);
    update(layer2States, layer2Params, layer2Grads, tid, stride, layer2Count);
    update(layer3States, layer3Params, layer3Grads, tid, stride, layer3Count);
    update(layer4States, layer4Params, layer4Grads, tid, stride, layer4Count);
}