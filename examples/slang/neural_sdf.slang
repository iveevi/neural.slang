import camera;
import sdf;
import mesh;
import shapes;
import random;

import neural;
import network_with_addresses;

// Rendering the reference mesh
ParameterBlock<TriangleMesh> mesh;

uniform RayFrame rayFrame;
uniform RWTexture2D targetTexture;
uniform uint2 targetResolution;

[shader("compute")]
[numthreads(32, 1, 1)]
void reference(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = (tid.xy + 0.5) / targetResolution;
    uv.y = 1 - uv.y;

    let ray = rayFrame.rayAt(uv);
    let sphere = Sphere(float3(0, 0, 0), 1);

    let intx = PositionNormalIntersection.sphereTrace((TriangleMesh) mesh, ray, 100, 1e-3, 10.0);
    if (intx.t > 0.0)
        targetTexture.Store(tid.xy, float4(intx.n * 0.5 + 0.5, 1));
    else
        targetTexture.Store(tid.xy, float4(0, 0, 0, 1));
}

// Rendering the neural SDF
extern static const int Hidden;
extern static const int HiddenLayers;
extern static const int Levels;

// typealias Encoder = FrequencyEncoder<float, 3, Levels>;
typealias Encoder = IdentityEncoder<float, 3>;

typealias Network = NetworkWithAddresses<
    3, 1, Encoder.Out,
    Hidden, HiddenLayers,
    Encoder,
    ReLU<float>,
>;

ParameterBlock<Network> network;

public extension Network : ISignedDistanceField
{
    public float sdf(float3 p)
    {
        return forward(Network.InputVec(float[3](p.x, p.y, p.z)))[0];
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void neural(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = (tid.xy + 0.5) / targetResolution;
    uv.y = 1 - uv.y;

    let ray = rayFrame.rayAt(uv);
    let oneAway = ray.origin + ray.direction;
    let sdf = network.sdf(oneAway);

    let intx = PositionNormalIntersection.sphereTrace((Network) network, ray, 100, 1e-3, 10.0);
    if (intx.t > 0.0)
        targetTexture.Store(tid.xy, float4(intx.n * 0.5 + 0.5, 1));
    else
        targetTexture.Store(tid.xy, float4(0, 0, 0, 1));
}

// Sampling the neural model
uniform RWStructuredBuffer<float3, ScalarDataLayout> sampleBuffer;
uniform RWStructuredBuffer<float> sdfBuffer;

[shader("compute")]
[numthreads(32, 1, 1)]
void sample_neural(uint3 tid : SV_DispatchThreadID)
{
    float3 sample = sampleBuffer[tid.x];
    let sdf = network.sdf(sample);
    sdfBuffer[tid.x] = sdf;
}

// Sampling the reference mesh
[shader("compute")]
[numthreads(32, 1, 1)]
void sample_reference(uint3 tid : SV_DispatchThreadID)
{
    float3 sample = sampleBuffer[tid.x];
    let sdf = mesh.sdf(sample);
    sdfBuffer[tid.x] = sdf;
}

// Evaluating gradients
uniform RWStructuredBuffer<float> lossBuffer;
uniform float lossBoost;
uniform uint time;

float loss<Object>(Network network, no_diff Object object, no_diff float3 sample)
    where Object : ISignedDistanceField
{
    static MeanSquaredError<float> mse;
    
    let meshSdf = object.sdf(sample);
    let networkSdf = network.sdf(sample);
    
    return mse.eval(
        Network.OutputVec(networkSdf),
        Network.OutputVec(meshSdf)
    );
}

[shader("compute")]
[numthreads(32, 1, 1)]
void gradient(uint3 tid : SV_DispatchThreadID)
{
    uint64_t seed = uint64_t(tid.x) << 32 | uint64_t(time);

    // PCG pcg = PCG(seed);

    // float3 sample = pcg.rand<3>();
    // sample = 2 * (2 * sample - 1);

    // sampleBuffer[tid.x] = sample;

    float3 sample = sampleBuffer[tid.x];
    
    Sphere object = Sphere(float3(0, 0, 0), 1);

    // TODO: needs proper SDF to work...
    // TriangleMesh object = mesh;
    // Box object = Box(float3(0, 0, 0), float3(1, 1, 1));

    let expected = object.sdf(sample);
    let lossValue = loss(network, object, sample);
    lossBuffer[tid.x] = lossValue;

    network.backward(
        Network.InputVec(float[3](sample.x, sample.y, sample.z)),
        Network.OutputVec(expected),
        lossBoost
    );
}

// Optimizing the network
[shader("compute")]
[numthreads(32, 1, 1)]
void optimize(uint3 tid : SV_DispatchThreadID)
{
    network.optimize(tid.x);
}