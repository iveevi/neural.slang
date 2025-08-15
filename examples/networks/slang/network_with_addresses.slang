module network_with_addresses;

import neural;

// Neural network structure
// TODO: INetwork interface with forward, backward, and optimize methods
public struct NetworkWithAddresses<
    int In,
    int Out,
    int EncodedOut,
    int Hidden,
    int HiddenLayers,
    Encoder,
    Activation
>
    where Encoder : IEncoder<float, In, EncodedOut>
    where Activation : IActivation<float>
{
    typealias StorageType = StructuredBufferStorage<float>;
    typealias StorageDiffType = DifferentialPtrPair<StorageType>;

    typealias FirstLayer = FeedForward<float, EncodedOut, Hidden, StorageType, Activation>;
    typealias HiddenLayer = FeedForward<float, Hidden, Hidden, StorageType, Activation>;
    typealias LastLayer = FeedForward<float, Hidden, Out, StorageType, Identity<float>>;

    typealias HiddenVec = InlineVector<float, Hidden>;
    typealias EncodedVec = InlineVector<float, EncodedOut>;

    public typealias InputVec = InlineVector<float, In>;
    public typealias OutputVec = InlineVector<float, Out>;

    // Network parameters and configuration
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> gradients;
    RWStructuredBuffer<Adam<float>.State> states;

    RWStructuredBuffer<int> layerAddresses;
    uint parameterCount;

    public typealias Parameters = StructuredBufferStorage<float>;
    public typealias Configuration = RWStructuredBuffer<int>;

    public Parameters parameterStorage()
    {
        return StructuredBufferStorage<float>(parameters);
    }

    public DifferentialPtrPair<StorageType> parameterStorageDiff()
    {
        let parameterStorage = StructuredBufferStorage<float>(parameters);
        let gradientStorage = StructuredBufferStorage<float>(gradients);
        return DifferentialPtrPair<StorageType>(parameterStorage, gradientStorage);
    }

    public Configuration addresses()
    {
        return layerAddresses;
    }

    // Evaluation and loss methods
    [BackwardDifferentiable]
    public static OutputVec eval(
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

    // TODO: this should also be generated from slangpy
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
    // TODO: becomes batch forward
    public OutputVec forward(InputVec input)
    {
        return eval(parameterStorage(), layerAddresses, input);
    }

    // TODO: becomes batch backward
    public void backward(InputVec input, OutputVec expected, float boost)
    {
        bwd_diff(loss)(
            parameterStorageDiff(),
            layerAddresses,
            input,
            expected,
            boost
        );
    }

    // TODO: optimize kernel can be generated from slangpy
    public void optimize(uint tid)
    {
        if (tid < parameterCount)
        {
            let optimizer = Adam<float>();
            optimizer.step(states[tid], parameters[tid], gradients[tid]);
            gradients[tid] = 0.0;
        }
    }
};
