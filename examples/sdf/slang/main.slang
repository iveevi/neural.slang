import camera;
import cmap;
import mesh;
import random;
import sdf;
import shapes;

import feature_grid;
import mlp;
import neural;
import network_with_addresses;

// Neural network and extensions
typealias MLP = AddressBasedMLP<3, 1, 32, 2, ReLU<float>, Identity<float>>;
typealias Optimizer = Adam<float>;

uniform MLP mlp;
// uniform FeatureGrid<3, 8> featureGrid;
uniform FeatureGrid<3, 1> sdfGrid;

public extension MLP : ISignedDistanceField
{
    public float sdf(float3 p)
    {
        let input = InlineVector<float, 3>.fromVector(p);
        // return sdfGrid.sample(sdfGrid.parameters, sdfGrid.configuration, input)[0];
        return eval(parameters, InputVec.fromVector(p))[0];
    }
}

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
    let intx = sphereTrace(mlp, ray, 100, 1e-3, 10.0);
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
    let intx = sphereTrace(mlp, ray, 100, 1e-3, 10.0);
    float3 color = 0.5 * intx.n + 0.5;
    targetTexture.Store(tid.xy, float4(color, 1));
}

// Backward pass
[BackwardDifferentiable]
float loss(
    mlp.Parameters parameters,
    sdfGrid.Parameters sdfGridParameters,
    no_diff float3 p,
    no_diff float s)
{
    static const let mse = MeanSquaredError<float>();
    
    let input = InlineVector<float, 3>.fromVector(p);
    let expected = InlineVector<float, 1>.fromVector(s);

    // let predicted = sdfGrid.sample(sdfGridParameters, sdfGrid.configuration, input);
    let predicted = MLP.eval(parameters, input);

    return mse.eval(predicted, expected);
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
        mlp.parameters,
        sdfGrid.parameters,
        inputBuffer[tid.x],
        expectedBuffer[tid.x]
    );

    bwd_diff(loss)(
        mlp.dual,
        sdfGrid.dual,
        inputBuffer[tid.x],
        expectedBuffer[tid.x],
        boost
    );
}

// Optimization
uniform Optimizer optimizer;
uniform RWStructuredBuffer<Optimizer.State> optimizerStates;
// TODO: one parameter block

[shader("compute")]
[numthreads(32, 1, 1)]
void optimize(uint3 tid : SV_DispatchThreadID)
{
    mlp.update(optimizer, optimizerStates[tid.x], tid.x);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void update_sdf_grid(uint3 tid : SV_DispatchThreadID)
{
    sdfGrid.update(optimizer, optimizerStates[tid.x], tid.x);
}