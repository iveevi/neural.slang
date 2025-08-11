module network_with_addresses;

import neural;

// Neural network structure
public struct NetworkWithAddresses<
    int In,
    int Out,
    int Hidden,
    int Levels,
    int HiddenLayers,
    Encoder,
    Activation
>
    where Encoder : IEncoder<float, In>
    where Activation : IActivation<float>
{
    typealias StorageType = StructuredBufferStorage<float>;
    typealias StorageDiffType = DifferentialPtrPair<StorageType>;

    typealias FirstLayer = FeedForward<float, Encoder.Out, Hidden, StorageType, Activation>;
    typealias HiddenLayer = FeedForward<float, Hidden, Hidden, StorageType, Activation>;
    typealias LastLayer = FeedForward<float, Hidden, Out, StorageType, Identity<float>>;

    typealias HiddenVec = InlineVector<float, Hidden>;
    typealias EncodedVec = InlineVector<float, Encoder.Out>;

    public typealias InputVec = InlineVector<float, In>;
    public typealias OutputVec = InlineVector<float, Out>;

    // Network parameters and configuration
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> gradients;
    RWStructuredBuffer<Adam<float>.State> states;

    RWStructuredBuffer<int> layerAddresses;
    uint parameterCount;

    // Evaluation and loss methods
    [BackwardDifferentiable]
    static OutputVec eval(
        StructuredBufferStorage<float> parameterStorage,
        no_diff RWStructuredBuffer<int> layerAddresses,
        no_diff InputVec input)
    {
        let encoder = Encoder();
        let encoded = encoder.eval<InputVec, EncodedVec>(input);

        let ff1 = FirstLayer(layerAddresses[0], {});
        var hidden = ff1.eval<EncodedVec, HiddenVec>(parameterStorage, encoded);

        [ForceUnroll]
        for (int i = 0; i < HiddenLayers; i++) {
            let ff = HiddenLayer(layerAddresses[i + 1], {});
            hidden = ff.eval<HiddenVec, HiddenVec>(parameterStorage, hidden);
        }

        let ffN = LastLayer(layerAddresses[HiddenLayers + 1], {});
        let output = ffN.eval<HiddenVec, OutputVec>(parameterStorage, hidden);

        return output;
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

    // Kernel methods
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
        let optimizer = Adam<float>();
        for (uint i = tid; i < parameterCount; i += dispatchSize) {
            optimizer.step(states[i], parameters[i], gradients[i]);
            gradients[i] = 0.0;
        }
    }
};