import camera;
import cmap;
import mesh;
import random;
import sdf;
import shapes;

import dense_grid;
import mlp;
import neural;
import network_with_addresses;

// Neural network and extensions
extern static const int Hidden = 64;
extern static const int HiddenLayers = 3;
extern static const int Features = 2;

typealias MLP = AddressBasedMLP<3 * Features, 1, Hidden, HiddenLayers, LeakyReLU<float>, Identity<float>>;
typealias Grid = DenseGrid<3, Features>;
typealias Optimizer = Adam<float>;

public struct MultiLevelDenseGrid
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
    public static InlineVector<float, 3 * Features> eval(StructuredBufferStorage<float> parameters, This configuration, InlineVector<float, 3> p)
    {
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
            Features,
        >(f0, f1);
        
        let f012 = concat<
            InlineVector<float, 3 * Features>,
            InlineVector<float, 2 * Features>,
            Grid.FeatureVector,
            float,
            2 * Features,
            Features,
        >(f01, f2);

        return f012;
    }
}

public struct Scene : IBoundedSignedDistanceField
{
    static const let box = Box(float3(-1, -1, -1), float3(1, 1, 1));
    
    MLP mlp;
    Grid grid;
    MultiLevelDenseGrid mldg;

    // TODO: parameter, configuration, etc. properties here as well

    public bool inBounds(float3 p)
    {
        return box.contains(p);
    }

    [NoDiffThis, Differentiable]
    public float3 normalized(float3 p)
    {
        return box.normalized(p);
    }

    public float sdf(float3 p)
    {
        // Only use sdf grid within bounding box
        if (inBounds(p))
        {
            let input = InlineVector<float, 3>.fromVector(normalized(p));
            // let feature = grid.eval(grid.parameters, grid.configuration, input);
            let feature = MultiLevelDenseGrid.eval(mldg.parameters, mldg.configuration, input);
            let predicted = mlp.eval(mlp.parameters, feature);
            return predicted[0];
        }
        else
        {
            return box.sdf(p);
        }
    }

    public float skip(Ray ray)
    {
        if (let intx = box.intersection(ray))
            return intx.t + 1e-3;
        else
            return 0.0;
    }
}

ParameterBlock<Scene> scene;

// Rendering the neural SDF
uniform RayFrame rayFrame;
uniform RWTexture2D targetTexture;
uniform uint2 targetResolution;

extern struct ColorMap : IColorMap = Plasma;

[shader("compute")]
[numthreads(32, 1, 1)]
void render_heatmap(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = (tid.xy + 0.5) / targetResolution;
    uv.y = 1 - uv.y;

    let ray = rayFrame.rayAt(uv);
    let intx = sphereTraceBounded((Scene) scene, ray, 100, 1e-3, 10.0);
    float3 color = ColorMap(0, 100).eval(intx.steps);
    targetTexture.Store(tid.xy, float4(color, 1));
}

[shader("compute")]
[numthreads(32, 1, 1)]
void render_normal(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = (tid.xy + 0.5) / targetResolution;
    uv.y = 1 - uv.y;

    let ray = rayFrame.rayAt(uv);
    let intx = sphereTraceBounded((Scene) scene, ray, 100, 1e-3, 10.0);
    float3 color = 0.5 * intx.n + 0.5;
    targetTexture.Store(tid.xy, float4(color, 1));
}

// TODO: eval method outside...

// Backward pass
[BackwardDifferentiable]
float loss(
    MLP.Parameters mlpParameters,
    Grid.Parameters gridParameters,
    StructuredBufferStorage<float> mldgParameters,
    no_diff float3 p,
    no_diff float s)
{
    let expected = InlineVector<float, 1>.fromVector(s);

    if (scene.inBounds(p))
    {
        let input = InlineVector<float, 3>.fromVector(scene.normalized(p));
        // let feature = scene.grid.eval(gridParameters, scene.grid.configuration, input);
        let feature = MultiLevelDenseGrid.eval(mldgParameters, scene.mldg.configuration, input);
        let predicted = scene.mlp.eval(mlpParameters, feature);
        return MSE<float>.eval(predicted, expected);
    }
    else
    {
        return 0.0;
    }
}

uniform RWStructuredBuffer<float3, ScalarDataLayout> inputBuffer;
uniform RWStructuredBuffer<float> expectedBuffer;
uniform RWStructuredBuffer<float> lossBuffer;
uniform float boost;

[shader("compute")]
[numthreads(32, 1, 1)]
void backward(uint3 tid : SV_DispatchThreadID)
{
    lossBuffer[tid.x] = loss(
        scene.mlp.parameters,
        scene.grid.parameters,
        scene.mldg.parameters,
        inputBuffer[tid.x],
        expectedBuffer[tid.x]
    );

    bwd_diff(loss)(
        scene.mlp.dual,
        scene.grid.dual,
        scene.mldg.dual,
        inputBuffer[tid.x],
        expectedBuffer[tid.x],
        boost
    );
}

// Optimization
uniform Optimizer optimizer;
uniform RWStructuredBuffer<Optimizer.State> optimizerStates;
// TODO: one parameter block

// TODO: general update method with extern struct for IOptimizable being updated...
[shader("compute")]
[numthreads(32, 1, 1)]
void update_mlp(uint3 tid : SV_DispatchThreadID)
{
    scene.mlp.update(optimizer, optimizerStates[tid.x], tid.x);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void update_grid(uint3 tid : SV_DispatchThreadID)
{
    scene.grid.update(optimizer, optimizerStates[tid.x], tid.x);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void update_multi_level_grid(uint3 tid : SV_DispatchThreadID)
{
    scene.mldg.update(optimizer, optimizerStates[tid.x], tid.x);
}