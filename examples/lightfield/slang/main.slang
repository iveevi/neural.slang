import camera;
import neural;
import network_with_addresses;
import shapes;
import cmap;

// Rendering the reference
struct VertexResult
{
    float4 position : SV_Position;
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

// typealias Encoder = FrequencyEncoder<float, 3, Levels>;
typealias Encoder = IdentityEncoder<float, 4>;

// TODO: IMLP<float, Encoder.Out, 3>
typealias Network = NetworkWithAddresses<
    Encoder.In, 3, Encoder.Out,
    Hidden, HiddenLayers,
    Encoder,
    ReLU<float>,
>;

uniform RayFrame rayFrame;
uniform RWTexture2D targetTexture;
uniform uint2 targetResolution;
uniform Cylinder cylinder;

ParameterBlock<Network> network;

[BackwardDifferentiable]
float3 render(
    Network.Parameters parameters,
    Network.Configuration configuration,
    no_diff float3 c,
    no_diff float3 d
)
{
    let normal = float3(
        sin(c.x),
        0.0,
        cos(c.x)
    );
    let dd = dot(normal, d);
    let cc = float4(
        dd,
        sin(c.y),
        cos(c.y),
        c.z
    );
    return Network.eval(
        parameters,
        configuration,
        Network.InputVec.fromVector(cc)
    ).toVector();
}

[shader("compute")]
[numthreads(32, 1, 1)]
void render_neural(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = (tid.xy + 0.5) / targetResolution;
    
    let ray = rayFrame.rayAt(uv);
    let intx = cylinder.intersection(ray);

    var color = float3(0.0, 0.0, 0.0);
    if (let x = intx)
    {
        let cylindricalCoords = cylinder.toCylindrical(x);
        color = render(network.parameterStorage(), network.addresses(), cylindricalCoords, ray.direction);
        // let ntheta = (cylindricalCoords.y + float.getPi()) / (2.0 * float.getPi());
        // let hsv = HSV(0.0, 1.0);
        // color = hsv.eval(cylindricalCoords.z);
    }

    targetTexture[tid.xy] = float4(color, 1.0);
}

// Backward pass
// TODO: multiple cameras at once...
uniform RWStructuredBuffer<float> lossBuffer;

[BackwardDifferentiable]
float loss(
    Network.Parameters parameters,
    Network.Configuration configuration,
    no_diff float3 cylindricalCoords,
    no_diff float3 direction,
    no_diff float3 expected
)
{
    let color = render(parameters, configuration, cylindricalCoords, direction);
    let mse = MeanSquaredError<float>();
    return mse.eval(
        Network.OutputVec.fromVector(color),
        Network.OutputVec.fromVector(expected),
    );
}

[shader("compute")]
[numthreads(32, 1, 1)]
void backward(uint3 tid : SV_DispatchThreadID)
{
    let uv = float2(tid.xy + 0.5) / targetResolution;

    let ray = rayFrame.rayAt(uv);
    let intx = cylinder.intersection(ray);

    if (!intx.hasValue)
        return;

    let cylindricalCoords = cylinder.toCylindrical(intx.value);
    let expected = targetTexture[tid.xy].rgb;

    bwd_diff(loss)(
        network.parameterStorageDiff(),
        network.addresses(),
        cylindricalCoords,
        ray.direction,
        expected,
        1.0
    );

    let index = tid.y * targetResolution.x + tid.x;
    
    lossBuffer[index] = loss(
        network.parameterStorage(),
        network.addresses(),
        cylindricalCoords,
        ray.direction,
        expected
    );
}