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
    public typealias Dual = DifferentialPtrPair<Parameters>;

    RWStructuredBuffer<float> parameterBuffer;
    RWStructuredBuffer<float> gradientBuffer;

    public property Parameters parameters {
        get {
            return StructuredBufferStorage<float>(parameterBuffer);
        }
    }

    public property Dual dual {
        get {
            let p = StructuredBufferStorage<float>(parameterBuffer);
            let g = StructuredBufferStorage<float>(gradientBuffer);
            return DifferentialPtrPair<Parameters>(p, g);
        }
    }

    public void update<Optimizer>(Optimizer optimizer, inout Optimizer.State state, uint tid)
        where Optimizer : IOptimizer<float>
    {
        optimizer.step(state, parameterBuffer[tid], gradientBuffer[tid]);
        gradientBuffer[tid] = 0.0;
    }

    [BackwardDifferentiable]
    public static OutputVec eval(Parameters parameterStorage, InputVec input)
    {
        typealias HiddenVec = InlineVector<float, Hidden>;

        typealias StorageType = StructuredBufferStorage<float>;
        typealias FirstLayer = FeedForward<float, In, Hidden, StorageType, HiddenActivation>;
        typealias HiddenLayer = FeedForward<float, Hidden, Hidden, StorageType, HiddenActivation>;
        typealias LastLayer = FeedForward<float, Hidden, Out, StorageType, OutputActivation>;

        var offset = 0;
        
        let ff1 = FirstLayer(offset, {});
        var hidden = ff1.eval<InputVec, HiddenVec>(parameterStorage, input);
        offset += (In + 1) * Hidden;

        [ForceUnroll]
        for (int i = 0; i < HiddenLayers; i++) {
            let ff = HiddenLayer(offset, {});
            hidden = ff.eval<HiddenVec, HiddenVec>(parameterStorage, hidden);
            offset += (Hidden + 1) * Hidden;
        }

        let ffN = LastLayer(offset, {});
        let output = ffN.eval<HiddenVec, OutputVec>(parameterStorage, hidden);

        return output;
    }
};
