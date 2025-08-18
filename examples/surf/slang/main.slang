import camera;
import neural;
import network_with_addresses;
import shapes;
import cmap;

import mlp;
import dense_grid;

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

// Rendering the neural network
extern static const int Hidden;
extern static const int HiddenLayers;
extern static const int Levels;

typealias MLP = AddressBasedMLP<8, 3, Hidden, HiddenLayers, LeakyReLU<float>, Identity<float>>;
typealias Grid1 = DenseGrid<2, 4>;
typealias Grid2 = DenseGrid<2, 4>;

uniform RayFrame rayFrame;
uniform RWTexture2D targetTexture;
uniform uint2 targetResolution;
uniform Cylinder cylinder;
uniform bool showHalf;
uniform float boost;

ParameterBlock<MLP> mlp;
ParameterBlock<Grid1> grid1;
ParameterBlock<Grid2> grid2;

[BackwardDifferentiable]
float3 render(
    MLP.Parameters parameters,
    Grid1.Parameters gridParameters1,
    Grid2.Parameters gridParameters2,
    no_diff CylinderCoordinates c1,
    no_diff CylinderCoordinates c2,
)
{
    var gridInput1 = InlineVector<float, 2>.fromVector(0.0);
    gridInput1[0] = c1.normalizedTheta();
    gridInput1[1] = c1.y;

    let feature1 = grid1.eval(gridParameters1, grid1.configuration, gridInput1);

    var gridInput2 = InlineVector<float, 2>.fromVector(0.0);
    gridInput2[0] = c2.normalizedTheta();
    gridInput2[1] = c2.y;

    let feature2 = grid2.eval(gridParameters2, grid2.configuration, gridInput2);

    // return feature1.slice<0, 3>().toVector().add(feature2.slice<0, 3>().toVector());

    let input = concat<
        InlineVector<float, 8>,
        InlineVector<float, 4>,
        InlineVector<float, 4>,
        float, 4, 4
    >(feature1, feature2);

    // let encoder = FrequencyEncoder<float, 4, 8>();

    // var combined = InlineVector<float, 4>(0.0);
    // combined[0] = 2.0 * c1.normalizedTheta();
    // combined[1] = 2.0 * c2.normalizedTheta();
    // combined[2] = c1.y;
    // combined[3] = c2.y;

    // let input = encoder.eval<
    //     InlineVector<float, 4>,
    //     InlineVector<float, 64>
    // >(combined);

    return MLP.eval(parameters, input).toVector();
}

struct DoubleIntersection
{
    CylinderCoordinates c1;
    CylinderCoordinates c2;
}

Optional<DoubleIntersection> intersections(Ray ray)
{
    let intx1 = cylinder.intersection(ray);
    if (!intx1.hasValue)
        return {};

    // From now always return a valid double intersection
    CylinderCoordinates c1;
    CylinderCoordinates c2;

    c1 = cylinder.toCylindrical(intx1.value.point);

    let intx2 = cylinder.intersection(ray);
    if (intx2.hasValue)
        c2 = cylinder.toCylindrical(intx2.value.point);
    else
        c2 = c1;

    return DoubleIntersection(c1, c2);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void render_neural(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = (tid.xy + 0.5) / targetResolution;
    uv.y = 1.0 - uv.y;  // Flip Y to match rasterization coordinate system
    float3 color = float3(0.0, 0.0, 0.0);

    if (showHalf && uv.x < 0.5)
        return;
    
    let ray = rayFrame.rayAt(uv);
    let dintx = intersections(ray);

    if (dintx.hasValue)
    {
        let c1 = dintx.value.c1;
        let c2 = dintx.value.c2;

        color = render(mlp.parameters, grid1.parameters, grid2.parameters, c1, c2);
    }
    else
    {
        color = float3(0.0, 0.0, 0.0);
    }

    targetTexture[tid.xy] = float4(color, 1.0);
}

// Backward pass
// TODO: multiple cameras at once...
uniform RWStructuredBuffer<float> lossBuffer;

[BackwardDifferentiable]
float loss(
    MLP.Parameters parameters,
    Grid1.Parameters gridParameters1,
    Grid2.Parameters gridParameters2,
    no_diff CylinderCoordinates c1,
    no_diff CylinderCoordinates c2,
    no_diff float3 expected
)
{
    let color = render(parameters, gridParameters1, gridParameters2, c1, c2);
    let mse = MeanSquaredError<float>();
    return mse.eval(
        MLP.OutputVec.fromVector(color),
        MLP.OutputVec.fromVector(expected),
    );
}

[shader("compute")]
[numthreads(32, 1, 1)]
void backward(uint3 tid : SV_DispatchThreadID)
{
    var uv = float2(tid.xy + 0.5) / targetResolution;
    uv.y = 1.0 - uv.y;  // Flip Y to match rasterization coordinate system

    let ray = rayFrame.rayAt(uv);
    let dintx = intersections(ray);

    if (!dintx.hasValue)
        return;

    let c1 = dintx.value.c1;
    let c2 = dintx.value.c2;

    let expected = targetTexture[tid.xy].rgb;

    bwd_diff(loss)(
        mlp.dual,
        grid1.dual,
        grid2.dual,
        c1,
        c2,
        expected,
        boost
    );

    let index = tid.y * targetResolution.x + tid.x;
    
    lossBuffer[index] = loss(
        mlp.parameters,
        grid1.parameters,
        grid2.parameters,
        c1,
        c2,
        expected
    );
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
void update_grid1(uint3 tid : SV_DispatchThreadID)
{
    if (isnan(grid1.gradients.buffer[tid.x]))
        return;

    grid1.update(optimizer, optimizerStates[tid.x], tid.x);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void update_grid2(uint3 tid : SV_DispatchThreadID)
{
    if (isnan(grid2.gradients.buffer[tid.x]))
        return;

    grid2.update(optimizer, optimizerStates[tid.x], tid.x);
}