implementing ngp;

public struct MultiLevelDenseGrid<int Features>
{
    RWStructuredBuffer<float> parameterBuffer;
    RWStructuredBuffer<float> gradientBuffer;
    Array<int, 3> resolutions;

    public property StructuredBufferStorage<float> parameters
    {
        get { return StructuredBufferStorage<float>(parameterBuffer); }
    }

    public property StructuredBufferStorage<float> gradients
    {
        get { return StructuredBufferStorage<float>(gradientBuffer); }
    }

    public property DifferentialPtrPair<StructuredBufferStorage<float>> dual
    {
        get { return DifferentialPtrPair<StructuredBufferStorage<float>>(parameters, gradients); }
    }

    public property This configuration
    {
        get { return this; }
    }

    public void update<Optimizer>(Optimizer optimizer, inout Optimizer.State state, int tid)
        where Optimizer : IOptimizer<float>
    {
        optimizer.step(state, parameterBuffer[tid], gradientBuffer[tid]);
        gradientBuffer[tid] = 0.0;
    }

    [BackwardDifferentiable]
    public static OutputVector eval<OutputVector>(StructuredBufferStorage<float> parameters, This configuration, InlineVector<float, 3> p)
        where OutputVector : IVector<float, 3 * Features>
    {
        // static_assert(OutputVector is InlineVector<float, 3 * Features>, "OutputVector must be InlineVector<float, 3 * Features>");

        typealias Grid = DenseGrid<3, Features>;
        
        let g0 = Grid.Impl(0, configuration.resolutions[0]);
        let f0 = g0.sample(parameters, p);

        let g1 = Grid.Impl(g0.stride, configuration.resolutions[1]);
        let f1 = g1.sample(parameters, p);

        let g2 = Grid.Impl(g0.stride + g1.stride, configuration.resolutions[2]);
        let f2 = g2.sample(parameters, p);

        let f01 = concat<
            InlineVector<float, 2 * Features>,
            Grid.FeatureVector,
            Grid.FeatureVector,
            float,
            Features,
            Features
        >(f0, f1);
        
        let f012 = concat<
            InlineVector<float, 3 * Features>,
            InlineVector<float, 2 * Features>,
            Grid.FeatureVector,
            float,
            2 * Features,
            Features
        >(f01, f2);

        return __slang_noop_cast<OutputVector>(f012);
    }
}