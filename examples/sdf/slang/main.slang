import camera;
import cmap;
import mesh;
import random;
import sdf;
import shapes;

import mlp;
import neural;
import network_with_addresses;

// Neural network and extensions
typealias MLP = AddressBasedMLP<3, 1, 32, 2, ReLU<float>, Identity<float>>;
typealias Optimizer = Adam<float>;

public extension MLP : ISignedDistanceField
{
    public float sdf(float3 p)
    {
        return eval(parameters, InputVec.fromVector(p))[0];
    }
}

uniform MLP mlp;

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
    MLP.Parameters parameters,
    no_diff MLP.InputVec input,
    no_diff MLP.OutputVec expected)
{
    return MeanSquaredError<float>().eval(MLP.eval(parameters, input), expected);
}

uniform RWStructuredBuffer<MLP.InputVec> inputBuffer;
uniform RWStructuredBuffer<MLP.OutputVec> expectedBuffer;
uniform float boost;

[shader("compute")]
[numthreads(32, 1, 1)]
void backward(uint3 tid : SV_DispatchThreadID)
{
    bwd_diff(loss)(
        mlp.dual,
        inputBuffer[tid.x],
        expectedBuffer[tid.x],
        boost
    );
}

// Optimization
uniform Optimizer optimizer;
uniform RWStructuredBuffer<Optimizer.State> optimizerStates;

[shader("compute")]
[numthreads(32, 1, 1)]
void optimize(uint3 tid : SV_DispatchThreadID)
{
    mlp.update(optimizer, optimizerStates[tid.x], tid.x);
}