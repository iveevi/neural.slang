import neural;

extern static const int In;
extern static const int Out;
extern static const int Hidden;
extern static const int Layers;
extern static const int Levels;

typealias InputVec = InlineVector<float, In>;
typealias OutputVec = InlineVector<float, Out>;
typealias HiddenVec = InlineVector<float, Hidden>;
typealias EncodedVec = InlineVector<float, 2 * In * Levels>;

typealias LayerAddresses = Array<int, 4>;

typealias StorageType = StructuredBufferStorage<float>;
typealias StorageDiffType = DifferentialPtrPair<StorageType>;
typealias ActivationType = ReLU<float>;
typealias OptimizerType = Adam<float>;
typealias Layer<int In, int Out> = FeedForward<float, In, Out, StorageType, ActivationType>;
typealias LastLayer<int In, int Out> = FeedForward<float, In, Out, StorageType, Identity<float>>;

// TODO: neural network struct with all the buffers

// Neural network parameters
[BackwardDifferentiable]
OutputVec eval(
    StructuredBufferStorage<float> parameterStorage,
    no_diff LayerAddresses layerAddresses,
    InputVec input)
{
    let ff2 = Layer<Hidden, Hidden>(layerAddresses[1], {});
    let ff3 = Layer<Hidden, Hidden>(layerAddresses[2], {});
    let ff4 = LastLayer<Hidden, Out>(layerAddresses[3], {});

    if (Levels == 0)
    {
        let ff1 = Layer<In, Hidden>(layerAddresses[0], {});
        let x1 = ff1.eval<InputVec, HiddenVec>(parameterStorage, input);
        let x2 = ff2.eval<HiddenVec, HiddenVec>(parameterStorage, x1);
        let x3 = ff3.eval<HiddenVec, HiddenVec>(parameterStorage, x2);
        let output = ff4.eval<HiddenVec, OutputVec>(parameterStorage, x3);
        return output;
    }
    else
    {
        let encoder = FrequencyEncoder<float, In, Levels>();
        let encoded = encoder.eval<InputVec, EncodedVec>(input);

        let ff1 = Layer<2 * In * Levels, Hidden>(layerAddresses[0], {});
        let x1 = ff1.eval<EncodedVec, HiddenVec>(parameterStorage, encoded);
        let x2 = ff2.eval<HiddenVec, HiddenVec>(parameterStorage, x1);
        let x3 = ff3.eval<HiddenVec, HiddenVec>(parameterStorage, x2);
        let output = ff4.eval<HiddenVec, OutputVec>(parameterStorage, x3);
        return output;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void forward(
    RWStructuredBuffer<float> parameters,
    RWStructuredBuffer<InputVec> inputBuffer,
    RWStructuredBuffer<OutputVec> outputBuffer,
    uniform LayerAddresses layerAddresses,
    uint3 thread_id : SV_DispatchThreadID)
{
    // TODO: pad up to 64 threads for saturated work groups
    int tid = thread_id.x;

    let parameterStorage = StructuredBufferStorage<float>(parameters);
    let input = InputVec(inputBuffer[tid]);
    let output = eval(parameterStorage, layerAddresses, input);
    outputBuffer[tid] = output;
}

// Neural network gradients
[BackwardDifferentiable]
float loss(
    StructuredBufferStorage<float> parameterStorage,
    no_diff LayerAddresses layerAddresses,
	InputVec input,
	no_diff OutputVec expected)
{
    let mse = MeanSquaredError<float>();
    let output = eval(parameterStorage, layerAddresses, input);
    return mse.eval(output, expected);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void backward(
    RWStructuredBuffer<float> parameters,
    RWStructuredBuffer<float> gradients,
	RWStructuredBuffer<InputVec> inputBuffer,
    RWStructuredBuffer<OutputVec> expectedBuffer,
	uint3 thread_id : SV_DispatchThreadID,
    uniform LayerAddresses layerAddresses,
    uniform float boost)
{
    int tid = thread_id.x;

    let parameterStorage = StructuredBufferStorage<float>(parameters);
    let gradientStorage = StructuredBufferStorage<float>(gradients);
    let diffStorage = DifferentialPtrPair<StorageType>(parameterStorage, gradientStorage);

    var dinput = diffPair(InputVec(inputBuffer[tid]));

    bwd_diff(loss)(
        diffStorage,
        layerAddresses,
        dinput,
        expectedBuffer[tid],
        boost
    );
}

// Optimization
[shader("compute")]
[numthreads(32, 1, 1)]
void optimize(
    RWStructuredBuffer<float> parameters,
    RWStructuredBuffer<float> gradients,
    RWStructuredBuffer<OptimizerType.State> states,
    uint3 thread_id : SV_DispatchThreadID,
    uniform uint dispatchSize,
    uniform uint count)
{
    uint tid = thread_id.x;
    uint stride = dispatchSize;

    let optimizer = OptimizerType();
    for (uint i = tid; i < count; i += stride) {
        optimizer.step(states[i], parameters[i], gradients[i]);
        gradients[i] = 0.0;
    }
}