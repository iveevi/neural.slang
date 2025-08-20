implementing ngp;

import neural;

public struct DenseGrid<int Dimension, int Features>
{
    public typealias Impl = DenseGridImpl<Dimension, Features>;
    public typealias FeatureVector = InlineVector<float, Features>;

    public typealias Parameters = StructuredBufferStorage<float>;
    public typealias Configuration = This;
    public typealias Dual = DifferentialPtrPair<Parameters>;

    RWStructuredBuffer<float> parameterBuffer;
    RWStructuredBuffer<float> gradientBuffer;

    int offset;
    int resolution;

    public property Parameters parameters
    {
        get {
            return StructuredBufferStorage<float>(parameterBuffer);
        }
    }

    // // TODO: temporary
    // public property StructuredBufferStorage<float> gradients
    // {
    //     get {
    //         return StructuredBufferStorage<float>(gradientBuffer);
    //     }
    // }

    public property Configuration configuration
    {
        get {
            return this;
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
    public static FeatureVector eval(Parameters parameters, no_diff Configuration configuration, InlineVector<float, Dimension> p)
    {
        return Impl(
            configuration.offset,
            configuration.resolution
        ) .sample(parameters, p);
    }
}