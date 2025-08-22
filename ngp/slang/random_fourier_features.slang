implementing ngp;

public struct RandomFourierFeatures<int Dim, int Features> : IOptimizable<float>
{
    public typealias FeatureVector = InlineVector<float, Features>;
    public typealias InputVector = InlineVector<float, Dim>;

    public typealias Parameters = StructuredBufferStorage<float>;
    public typealias Dual = DifferentialPtrPair<StructuredBufferStorage<float>>;

    public static const int In = Dim;
    public static const int Out = Features;

    RWStructuredBuffer<float> parameterBuffer;
    RWStructuredBuffer<float> gradientBuffer;

    public property Parameters parameters
    {
        get {
            return StructuredBufferStorage<float>(parameterBuffer);
        }
    }
    
    public property Dual dual
    {
        get {
            let p = StructuredBufferStorage<float>(parameterBuffer);
            let g = StructuredBufferStorage<float>(gradientBuffer);
            return DifferentialPtrPair<StructuredBufferStorage<float>>(p, g);
        }
    }

    public void update<Optimizer>(Optimizer optimizer, inout Optimizer.State state, uint tid)
        where Optimizer : IOptimizer<float>
    {
        optimizer.step(state, parameterBuffer[tid], gradientBuffer[tid]);
        gradientBuffer[tid] = 0.0;
    }

    [BackwardDifferentiable]
    public FeatureVector eval(Parameters parameters, InputVector input)
    {
        let encoder = RandomFourierEncoder<float, Dim, Features, StructuredBufferStorage<float>>(0);
        return encoder.eval<InputVector, FeatureVector>(parameters, input);
    }
}