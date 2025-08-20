import camera;
import neural;
import ngp;
import shapes;
import cmap;

extern static const int Hidden;
extern static const int HiddenLayers;

// Rendering the reference
struct VertexResult
{
    float4 position : SV_Position;
    float3 worldPosition : POSITION;
    float3 worldNormal : NORMAL;
    float2 uv : TEXCOORD0;
}

uniform float4x4 view;
uniform float4x4 perspective;
uniform Texture2D diffuseTexture;
uniform SamplerState diffuseSampler;

[shader("vertex")]
VertexResult reference_vertex(float3 position : POSITION, float3 normal : NORMAL, float2 uv : TEXCOORD0)
{
    VertexResult result;

    float4 hom = float4(position, 1.0);
    float4 view_pos = mul(view, hom);
    result.position = mul(perspective, view_pos);
    
    result.worldPosition = position;
    result.worldNormal = normal;
    result.uv = uv;
    
    return result;
}

uniform RaytracingAccelerationStructure blas;

float rand(inout float seed)
{
    static const let sqrt2 = sqrt(2.0);
    seed = fmod(seed * sqrt2, 1.0);
    return seed;
}

float2 rand2(inout float seed)
{
    return float2(rand(seed), rand(seed));
}

struct AreaLight
{
    float3 center;
    float3 horizontal;
    float3 vertical;

    float3 sample(float2 uv)
    {
        return center + (uv.x - 0.5) * horizontal + (uv.y - 0.5) * vertical;
    }

    float3 color(float2 uv)
    {
        // Blue on the left, orange on the right
        if (uv.x < 0.5)
            return float3(0.23, 0.23, 0.80) * 2;
        else
            return float3(0.98, 0.47, 0.13) * 2;
    }
}

// TODO: use PRG
uniform Texture2D seedTexture;
uniform SamplerState seedSampler;
uniform float time;

[shader("fragment")]
float4 reference_fragment(VertexResult input) : SV_Target
{
    float seed = fmod(seedTexture.Sample(seedSampler, input.uv).r + 13.4234 * time, 1.0);
    float3 albedo = diffuseTexture.Sample(diffuseSampler, input.uv).rgb;
    float3 normal = normalize(input.worldNormal);
    
    // Area light from the top
    let light = AreaLight(
        float3(0.0, 7.0, 0.0),
        float3(1.0, 0.0, 0.0) * 5,
        float3(0.0, 0.0, 1.0) * 5,
    );

    // Ray casting for shadows
    let offset = input.worldPosition + 1e-3 * normal;
    let uv = rand2(seed);
    let sample = light.sample(uv);

    RayDesc ray;
    ray.Origin = offset;
    ray.Direction = normalize(sample - offset);
    ray.TMin = 1e-3;
    ray.TMax = 10.0;

    // Ray tracing for shadows
    RayQuery<
        RAY_FLAG_FORCE_OPAQUE
        | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES
        | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
    > query;

    query.TraceRayInline(blas, 0, 0xff, ray);
    query.Proceed();

    float shadow = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : 1.0;
    float ndotl = max(dot(normal, ray.Direction), 0.0);

    // TODO: try multiple bounces... also a bigger scene?

    return float4(float3(shadow) * light.color(uv) * albedo * ndotl, 1.0);
}

struct GBuffer
{
    float3 position : SV_Target0;
    float3 albedo : SV_Target1;
}

[shader("fragment")]
GBuffer gbuffer_fragment(VertexResult input) : SV_Target
{
    GBuffer result;
    result.position = input.worldPosition;
    result.albedo = diffuseTexture.Sample(diffuseSampler, input.uv).rgb;
    return result;
}

// TODO: multilevel grid...
static const let bound = Box(float3(-1.0), float3(1.0));

ParameterBlock<MLP> mlp;
ParameterBlock<Grid> grid;

typealias MLP = AddressBasedMLP<8, 3, Hidden, HiddenLayers, LeakyReLU<float>, Identity<float>>;
typealias Grid = DenseGrid<3, 8>;

[BackwardDifferentiable]
float3 shade(
    Grid.Parameters gridParameters,
    MLP.Parameters mlpParameters,
    no_diff float3 position,
    no_diff float3 albedo
)
{
    let input = InlineVector<float, 3>.fromVector(bound.normalized(position));
    let feature = Grid.eval(gridParameters, grid.configuration, input);
    let output = MLP.eval(mlpParameters, feature);
    return output.toVector() * albedo;
}

uniform RWTexture2D targetTexture;
uniform Texture2D gbufferPositionTexture;
uniform Texture2D gbufferAlbedoTexture;

[shader("compute")]
[numthreads(8, 8, 1)]
void deferred_neural_shading(uint3 tid : SV_DispatchThreadID)
{
    let position = gbufferPositionTexture[tid.xy];
    let albedo = gbufferAlbedoTexture[tid.xy];

    float3 color = float3(0.0, 0.0, 0.0);
    if (position.w < 1.0)
    {
        color = shade(grid.parameters, mlp.parameters, position.xyz, albedo.xyz);
    }

    targetTexture[tid.xy] = float4(color, 1.0);
}

[BackwardDifferentiable]
float loss(
    Grid.Parameters gridParameters,
    MLP.Parameters mlpParameters,
    no_diff float3 position,
    no_diff float3 albedo,
    no_diff float3 expected
)
{
    let output = shade(gridParameters, mlpParameters, position, albedo);
    let mse = MeanSquaredError<float>();
    let outputVec = InlineVector<float, 3>.fromVector(output);
    let expectedVec = InlineVector<float, 3>.fromVector(expected);
    return mse.eval(outputVec, expectedVec);
}

uniform Texture2D referenceTexture;
uniform float boost;

[shader("compute")]
[numthreads(8, 8, 1)]
void backward(uint3 tid : SV_DispatchThreadID)
{
    let position = gbufferPositionTexture[tid.xy];
    let albedo = gbufferAlbedoTexture[tid.xy];
    if (position.w < 1.0)
    {
        let expected = referenceTexture[tid.xy].rgb;
        bwd_diff(loss)(
            grid.dual,
            mlp.dual,
            position.xyz,
            albedo.xyz,
            expected,
            boost
        );
    }
}