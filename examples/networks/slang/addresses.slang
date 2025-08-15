implementing mlp;

public struct AddressBasedMLP<
    int In,
    int Out,
    int Hidden,
    int HiddenLayers,
    HiddenActivation,
    OutputActivation
> : IMLP<float, In, Out>
    where HiddenActivation : IActivation<float>
    where OutputActivation : IActivation<float>
{
    public typealias InputVec = InlineVector<float, In>;
    public typealias OutputVec = InlineVector<float, Out>;

    public typealias Parameters = StructuredBufferStorage<float>;
    public typealias Configuration = RWStructuredBuffer<int>;
    public typealias Dual = DifferentialPtrPair<Parameters>;

    RWStructuredBuffer<float>.Handle parameterBuffer;
    RWStructuredBuffer<float>.Handle gradientBuffer;
    RWStructuredBuffer<int>.Handle layerAddressBuffer;
    uint parameterCount;

    public property Parameters parameters {
        get {
            return StructuredBufferStorage<float>(parameterBuffer);
        }
    }

    public property Configuration configuration {
        get {
            return RWStructuredBuffer<int>(layerAddressBuffer);
        }
    }

    public property Dual dual {
        get {
            return DifferentialPtrPair<Parameters>(StructuredBufferStorage<float>(parameterBuffer), StructuredBufferStorage<float>(gradientBuffer));
        }
    }

    [BackwardDifferentiable]
    public static OutputVec eval(
        Parameters parameterStorage,
        Configuration layerAddresses,
        InputVec input)
    {
        typealias HiddenVec = InlineVector<float, Hidden>;

        typealias StorageType = StructuredBufferStorage<float>;
        typealias FirstLayer = FeedForward<float, In, Hidden, StorageType, HiddenActivation>;
        typealias HiddenLayer = FeedForward<float, Hidden, Hidden, StorageType, HiddenActivation>;
        typealias LastLayer = FeedForward<float, Hidden, Out, StorageType, OutputActivation>;

        let ff1 = FirstLayer(layerAddresses[0], {});
        var hidden = ff1.eval<InputVec, HiddenVec>(parameterStorage, input);

        [ForceUnroll]
        for (int i = 0; i < HiddenLayers; i++) {
            let ff = HiddenLayer(layerAddresses[i + 1], {});
            hidden = ff.eval<HiddenVec, HiddenVec>(parameterStorage, hidden);
        }

        let ffN = LastLayer(layerAddresses[HiddenLayers + 1], {});
        let output = ffN.eval<HiddenVec, OutputVec>(parameterStorage, hidden);

        return output;
    }
};
