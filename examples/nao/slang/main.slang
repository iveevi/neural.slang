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

// TODO: multilevel grid...
static const let bound = Box(float3(-1.0), float3(1.0));

ParameterBlock<MLP> mlp;
ParameterBlock<Grid> grid;

[BackwardDifferentiable]
InlineVector<float, 3> shade(Grid.Parameters gridParameters, float3 position)
{
    let input = InlineVector<float, 3>.fromVector(bound.normalized(position));
    let output = Grid.eval(gridParameters, grid.configuration, input);
    return output;
}

[shader("fragment")]
float4 render_neural_shading(VertexResult input) : SV_Target
{
    let shaded = shade(grid.parameters, input.worldPosition);
    return float4(shaded.toVector(), 1.0);
}

uniform Texture2D referenceTexture;
uniform RWStructuredBuffer<float> lossBuffer;
uniform RWStructuredBuffer<float> validBuffer;
uniform SamplerState referenceSampler;
uniform float2 targetResolution;
uniform float boost;

[BackwardDifferentiable]
float loss(
    Grid.Parameters gridParameters,
    no_diff float3 position,
    no_diff float3 expected
)
{
    let output = shade(gridParameters, position);
    let mse = MeanSquaredError<float>();
    let expectedVec = InlineVector<float, 3>.fromVector(expected);
    return mse.eval(output, expectedVec);
}

[shader("fragment")]
float4 render_backward(VertexResult input) : SV_Target
{
    float2 screenPos = input.position.xy;
    float2 uv = screenPos / targetResolution;
    float3 expected = referenceTexture.Sample(referenceSampler, uv).rgb;

    bwd_diff(loss)(
        grid.dual,
        input.worldPosition,
        expected,
        boost
    );

    let lossIndex = int(screenPos.y * targetResolution.x + screenPos.x);
    lossBuffer[lossIndex] = loss(
        grid.parameters,
        input.worldPosition,
        expected
    );

    validBuffer[lossIndex] = 1.0;
    
    return float4(0.0, 0.0, 0.0, 0.0);
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