import camera;
import neural;
import network_with_addresses;

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

typealias Encoder = FrequencyEncoder<float, 6, Levels>;
// typealias Encoder = IdentityEncoder<float, 6>;

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

ParameterBlock<Network> network;

[BackwardDifferentiable]
float3 render(
    Network.Parameters parameters,
    Network.Configuration configuration,
    no_diff Ray ray
)
{
    let p = InlineVector<float, 3>.fromVector(ray.origin);
    let d = InlineVector<float, 3>.fromVector(ray.direction);
    let v = concat<
        Network.InputVec,
        InlineVector<float, 3>,
        InlineVector<float, 3>,
        float, 3, 3
    >(p, d);
    return Network.eval(parameters, configuration, v).toVector();
}

[shader("compute")]
[numthreads(32, 1, 1)]
void render_neural(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = (tid.xy + 0.5) / targetResolution;
    let ray = rayFrame.rayAt(uv);
    let color = render(network.parameterStorage(), network.addresses(), ray);
    targetTexture[tid.xy] = float4(color, 1.0);
}

// Backward pass
uniform RWStructuredBuffer<uint2> pixelSamples;
uniform RWStructuredBuffer<float> lossBuffer;

[BackwardDifferentiable]
float loss(
    Network.Parameters parameters,
    Network.Configuration configuration,
    no_diff Ray ray,
    no_diff float3 expected
)
{
    let color = render(parameters, configuration, ray);
    let mse = MeanSquaredError<float>();
    return mse.eval(
        Network.OutputVec.fromVector(color),
        Network.OutputVec.fromVector(expected)
    );
}

[shader("compute")]
[numthreads(32, 1, 1)]
void backward(uint3 tid : SV_DispatchThreadID)
{
    let pixel = pixelSamples[tid.x];
    let uv = float2(pixel) / targetResolution;
    let ray = rayFrame.rayAt(uv);

    let expected = targetTexture[pixel].rgb;

    bwd_diff(loss)(
        network.parameterStorageDiff(),
        network.addresses(),
        ray,
        expected,
        1.0
    );
    
    lossBuffer[tid.x] = loss(
        network.parameterStorage(),
        network.addresses(),
        ray,
        expected
    );
}

// TODO: separate MLP and encoder