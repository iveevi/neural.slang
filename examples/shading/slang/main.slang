import camera;
import neural;
import network_with_addresses;
import shapes;
import cmap;

import mlp;
import dense_grid;

extern static const int Hidden;
extern static const int HiddenLayers;
extern static const int Levels;

typealias MLP = AddressBasedMLP<3, 3, Hidden, HiddenLayers, LeakyReLU<float>, Identity<float>>;
typealias Grid = DenseGrid<3, 3>;

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

[shader("fragment")]
float4 reference_fragment(VertexResult input) : SV_Target
{
    float3 albedo = diffuseTexture.Sample(diffuseSampler, input.uv).rgb;
    
    // Simple directional light
    float3 lightDir = normalize(float3(1.0, 1.0, 1.0));
    float3 normal = normalize(input.worldNormal);
    
    // Lambertian shading
    float ndotl = max(dot(normal, lightDir), 0.0);
    float3 ambient = albedo * 0.1;
    float3 diffuse = albedo * ndotl * 0.9;
    
    return float4(ambient + diffuse, 1.0);
}

struct GBuffer
{
    float3 position : SV_Target0;
}

[shader("fragment")]
GBuffer gbuffer_fragment(VertexResult input) : SV_Target
{
    GBuffer result;
    result.position = input.worldPosition;
    return result;
}

// TODO: multilevel grid...
static const let bound = Box(float3(-1.0), float3(1.0));

ParameterBlock<MLP> mlp;
ParameterBlock<Grid> grid;

[BackwardDifferentiable]
InlineVector<float, 3> shade(
    Grid.Parameters gridParameters,
    MLP.Parameters mlpParameters,
    float3 position
)
{
    let input = InlineVector<float, 3>.fromVector(bound.normalized(position));
    let feature = Grid.eval(gridParameters, grid.configuration, input);
    let output = MLP.eval(mlpParameters, feature);
    return output;
}

uniform RWTexture2D targetTexture;
uniform Texture2D gbufferPositionTexture;

[shader("compute")]
[numthreads(8, 8, 1)]
void deferred_neural_shading(uint3 tid : SV_DispatchThreadID)
{
    let position = gbufferPositionTexture[tid.xy];

    float3 color = float3(0.0, 0.0, 0.0);
    if (position.w < 1.0)
    {
        let shaded = shade(grid.parameters, mlp.parameters, position.xyz);
        color = shaded.toVector();
    }

    targetTexture[tid.xy] = float4(color, 1.0);
}

[BackwardDifferentiable]
float loss(
    Grid.Parameters gridParameters,
    MLP.Parameters mlpParameters,
    no_diff float3 position,
    no_diff float3 expected
)
{
    let output = shade(gridParameters, mlpParameters, position);
    let mse = MeanSquaredError<float>();
    let expectedVec = InlineVector<float, 3>.fromVector(expected);
    return mse.eval(output, expectedVec);
}

uniform Texture2D referenceTexture;
uniform float boost;

[shader("compute")]
[numthreads(8, 8, 1)]
void backward(uint3 tid : SV_DispatchThreadID)
{
    let position = gbufferPositionTexture[tid.xy];
    if (position.w < 1.0)
    {
        let expected = referenceTexture[tid.xy].rgb;
        bwd_diff(loss)(
            grid.dual,
            mlp.dual,
            position.xyz,
            expected,
            boost
        );
    }
}

// Optimization
typealias Optimizer = Adam<float>;

uniform Optimizer optimizer;
uniform RWStructuredBuffer<Optimizer.State> optimizerStates;

[shader("compute")]
[numthreads(32, 1, 1)]
void update_mlp(uint3 tid : SV_DispatchThreadID)
{
    mlp.update(optimizer, optimizerStates[tid.x], tid.x);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void update_grid(uint3 tid : SV_DispatchThreadID)
{
    if (isnan(grid.gradients.buffer[tid.x]))
        return;

    grid.update(optimizer, optimizerStates[tid.x], tid.x);
}