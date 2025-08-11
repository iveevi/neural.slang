import neural;

extern static const int In;
extern static const int Out;
extern static const int Hidden;
extern static const int HiddenLayers;
extern static const int Levels;

typealias OptimizerType = Adam<float>;

public struct NetworkWithAddresses<int In, int Out, int Hidden, int Levels, int HiddenLayers, ActivationType>
    where ActivationType : IActivation<float>
{
    typealias StorageType = StructuredBufferStorage<float>;
    typealias StorageDiffType = DifferentialPtrPair<StorageType>;

    typealias Layer<int In, int Out> = FeedForward<float, In, Out, StorageType, ActivationType>;
    typealias LastLayer<int In, int Out> = FeedForward<float, In, Out, StorageType, Identity<float>>;

    typealias HiddenVec = InlineVector<float, Hidden>;
    typealias EncodedVec = InlineVector<float, 2 * In * Levels>;

    public typealias InputVec = InlineVector<float, In>;
    public typealias OutputVec = InlineVector<float, Out>;

    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> gradients;
    RWStructuredBuffer<OptimizerType.State> states;

    RWStructuredBuffer<int> layerAddresses;
    uint parameterCount;

    [BackwardDifferentiable]
    static OutputVec eval(
        StructuredBufferStorage<float> parameterStorage,
        no_diff RWStructuredBuffer<int> layerAddresses,
        no_diff InputVec input)
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

    [BackwardDifferentiable]
    static float loss(
        StructuredBufferStorage<float> parameterStorage,
        no_diff RWStructuredBuffer<int> layerAddresses,
        no_diff InputVec input,
        no_diff OutputVec expected)
    {
        let mse = MeanSquaredError<float>();
        let output = eval(parameterStorage, layerAddresses, input);
        return mse.eval(output, expected);
    }

    public OutputVec forward(InputVec input)
    {
        let parameterStorage = StructuredBufferStorage<float>(parameters);
        return eval(parameterStorage, layerAddresses, input);
    }

    public void backward(InputVec input, OutputVec expected, float boost)
    {
        let parameterStorage = StructuredBufferStorage<float>(parameters);
        let gradientStorage = StructuredBufferStorage<float>(gradients);
        let diffStorage = DifferentialPtrPair<StorageType>(parameterStorage, gradientStorage);

        bwd_diff(loss)(
            diffStorage,
            layerAddresses,
            input,
            expected,
            boost
        );
    }

    public void optimize(uint tid, uint dispatchSize)
    {
        let optimizer = OptimizerType();
        for (uint i = tid; i < parameterCount; i += dispatchSize) {
            optimizer.step(states[i], parameters[i], gradients[i]);
            gradients[i] = 0.0;
        }
    }
};

typealias Network = NetworkWithAddresses<In, Out, Hidden, Levels, HiddenLayers, ReLU<float>>;
typealias InputVec = Network.InputVec;
typealias OutputVec = Network.OutputVec;

// uniform Network network;

// Neural network inference
[shader("compute")]
[numthreads(32, 1, 1)]
void forward(
    uniform Network network,
    RWStructuredBuffer<InputVec> inputBuffer,
    RWStructuredBuffer<OutputVec> outputBuffer,
    uint3 thread_id : SV_DispatchThreadID)
{
    // TODO: pad up to 64 threads for saturated work groups
    int tid = thread_id.x;

    let input = InputVec(inputBuffer[tid]);
    let output = network.forward(input);
    outputBuffer[tid] = output;
}

// Neural network training
[shader("compute")]
[numthreads(32, 1, 1)]
void backward(
    uniform Network network,
	RWStructuredBuffer<InputVec> inputBuffer,
    RWStructuredBuffer<OutputVec> expectedBuffer,
	uint3 thread_id : SV_DispatchThreadID,
    uniform float boost)
{
    int tid = thread_id.x;

    let input = InputVec(inputBuffer[tid]);
    let expected = OutputVec(expectedBuffer[tid]);

    network.backward(input, expected, boost);
}

// Neural network optimization
[shader("compute")]
[numthreads(32, 1, 1)]
void optimize(
    uniform Network network,
    uint3 thread_id : SV_DispatchThreadID,
    uniform uint dispatchSize)
{
    network.optimize(thread_id.x, dispatchSize);
}