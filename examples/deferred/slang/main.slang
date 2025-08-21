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
uniform float2 targetResolution;

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

float3 rotate(float3 s, float3 n)
{
    static const let epsilon = 1e-6;
    let w = -n;
    let u = normalize(abs(w.z) < (1 - epsilon) ? float3(-w.y, w.x, 0) : float3(0, -w.z, w.y));
    let v = cross(w, u);
    return s.x * u + s.y * v + s.z * w;
}

uniform Texture2D seedTexture;
uniform SamplerState seedSampler;
uniform float time;

[shader("fragment")]
float4 reference_fragment(VertexResult input, bool isFrontFace : SV_IsFrontFace) : SV_Target
{
    float2 coord = input.position.xy / targetResolution;
    float blueNoise = seedTexture.Sample(seedSampler, coord).r;
    float seed = frac(blueNoise * time);

    float3 albedo = diffuseTexture.Sample(diffuseSampler, input.uv).rgb;
    float3 normal = normalize(input.worldNormal);
    if (!isFrontFace)
        normal = -normal;
    
    // Random hemisphere sampling aligned with the normal
    let r1 = rand(seed);
    let r2 = rand(seed);
    let theta = 2.0 * float.getPi() * r1;
    let phi = acos(2.0 * r2 - 1.0);
    let v = float3(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
    let direction = rotate(v, normal);
    
    // Lighting from a far away sun
    let sunDirection = float3(1.0, 1.0, 0.0);
    let ldot = max(dot(direction, sunDirection), 0.0);
    let color = pow(ldot, 4.0) * float3(1.0, 1.0, 0.5);

    // Ray tracing for shadows
    RayDesc ray;
    ray.Origin = input.worldPosition + 1e-3 * normal;
    ray.Direction = normalize(direction);
    ray.TMin = 1e-3;
    ray.TMax = 10.0;

    RayQuery<
        RAY_FLAG_FORCE_OPAQUE
        | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES
        | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
    > query;

    query.TraceRayInline(blas, 0, 0xff, ray);
    query.Proceed();

    float shadow = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : 1.0;

    // Final shading
    float ndotl = max(dot(normal, ray.Direction), 0.0);
    float3 finalColor = color * albedo * ndotl;
    return float4(finalColor, 1.0);
}

struct GBuffer
{
    float3 position : SV_Target0;
    float3 albedo : SV_Target1;
    float3 normal : SV_Target2;
}

[shader("fragment")]
GBuffer gbuffer_fragment(VertexResult input, bool isFrontFace : SV_IsFrontFace) : SV_Target
{
    GBuffer result;
    result.position = input.worldPosition;
    result.albedo = diffuseTexture.Sample(diffuseSampler, input.uv).rgb;
    result.normal = isFrontFace ? input.worldNormal : -input.worldNormal;
    return result;
}

// TODO: multilevel grid...
static const let bound = Box(float3(-1.0), float3(1.0));

ParameterBlock<MLP> mlp;
ParameterBlock<Grid> grid;

typealias MLP = AddressBasedMLP<11, 3, Hidden, HiddenLayers, LeakyReLU<float>, Identity<float>>;
typealias Grid = DenseGrid<3, 8>;

[BackwardDifferentiable]
float3 illumination(
    Grid.Parameters gridParameters,
    MLP.Parameters mlpParameters,
    no_diff float3 position,
    no_diff float3 normal
)
{
    let input = InlineVector<float, 3>.fromVector(bound.normalized(position));
    let feature = Grid.eval(gridParameters, grid.configuration, input);
    let nvec = InlineVector<float, 3>.fromVector(normal);
    let concatted = concat<
        InlineVector<float, 11>,
        InlineVector<float, 8>,
        InlineVector<float, 3>,
        float, 8, 3
    >(feature, nvec);
    let output = MLP.eval(mlpParameters, concatted);
    return output.toVector();
}

[BackwardDifferentiable]
float3 shade(
    Grid.Parameters gridParameters,
    MLP.Parameters mlpParameters,
    no_diff float3 position,
    no_diff float3 albedo,
    no_diff float3 normal
)
{
    return illumination(gridParameters, mlpParameters, position, normal) * albedo;
}

uniform RWTexture2D targetTexture;
uniform Texture2D gbufferPositionTexture;
uniform Texture2D gbufferAlbedoTexture;
uniform Texture2D gbufferNormalTexture;

[shader("compute")]
[numthreads(8, 8, 1)]
void deferred_neural_shading(uint3 tid : SV_DispatchThreadID)
{
    let position = gbufferPositionTexture[tid.xy];
    let albedo = gbufferAlbedoTexture[tid.xy];
    let normal = normalize(gbufferNormalTexture[tid.xy].xyz);

    float3 color = float3(0.0, 0.0, 0.0);
    if (position.w < 1.0)
        color = shade(grid.parameters, mlp.parameters, position.xyz, albedo.xyz, normal);

    targetTexture[tid.xy] = float4(color, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void deferred_neural_illumination(uint3 tid : SV_DispatchThreadID)
{
    let position = gbufferPositionTexture[tid.xy];
    let albedo = gbufferAlbedoTexture[tid.xy];
    let normal = normalize(gbufferNormalTexture[tid.xy].xyz);

    float3 color = float3(0.0, 0.0, 0.0);
    if (position.w < 1.0)
        color = illumination(grid.parameters, mlp.parameters, position.xyz, normal);

    targetTexture[tid.xy] = float4(color, 1.0);
}

[BackwardDifferentiable]
float loss(
    Grid.Parameters gridParameters,
    MLP.Parameters mlpParameters,
    no_diff float3 position,
    no_diff float3 albedo,
    no_diff float3 normal,
    no_diff float3 expected
)
{
    let output = shade(gridParameters, mlpParameters, position, albedo, normal);
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
    let normal = normalize(gbufferNormalTexture[tid.xy].xyz);
    
    if (position.w < 1.0)
    {
        let expected = referenceTexture[tid.xy].rgb;
        bwd_diff(loss)(
            grid.dual,
            mlp.dual,
            position.xyz,
            albedo.xyz,
            normal,
            expected,
            boost
        );
    }
}

uniform RWStructuredBuffer<float> stableParameters;
uniform RWStructuredBuffer<float> trainedParameters;
uniform float eta;

[shader("compute")]
[numthreads(32, 1, 1)]
void update_stable(uint3 tid : SV_DispatchThreadID)
{
    let stable = stableParameters[tid.x];
    let trained = trainedParameters[tid.x];
    let updated = lerp(stable, trained, eta);
    stableParameters[tid.x] = updated;
}