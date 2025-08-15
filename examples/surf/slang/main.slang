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
typealias Encoder = IdentityEncoder<float, 7>;

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
    no_diff float3 rayDirection
)
{
    // c contains cylindrical coordinates: (r, theta, y)
    // where r is normalized radius, theta is angle, y is normalized height
    
    // Calculate surface normal at intersection point (outward pointing)
    // For a cylinder, the normal is radial in the xz plane
    let surfaceNormal = float3(cos(c.y), 0.0, sin(c.y));
    
    // Create normal frame at intersection point
    // Normal: outward radial direction
    // Tangent: tangent to cylinder surface (perpendicular to normal in xz plane)
    // Bitangent: vertical direction (y-axis)
    let normal = surfaceNormal;
    let tangent = float3(-sin(c.y), 0.0, cos(c.y));  // perpendicular to normal in xz plane
    let bitangent = float3(0.0, 1.0, 0.0);  // vertical direction
    
    // Invert the ray direction (from intersection point toward camera)
    let invertedRay = -rayDirection;
    
    // Transform inverted ray to normal frame coordinates
    let rayInNormalFrame = float3(
        dot(invertedRay, tangent),
        dot(invertedRay, bitangent), 
        dot(invertedRay, normal)
    );
    
    // Calculate spherical angles in normal frame
    // Polar angle: angle from normal (z-axis in normal frame)
    let cosTheta = rayInNormalFrame.z / length(rayInNormalFrame);
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    
    // Azimuthal angle: angle around normal in tangent-bitangent plane
    let cosPhi = rayInNormalFrame.x / length(rayInNormalFrame.xy);
    let sinPhi = rayInNormalFrame.y / length(rayInNormalFrame.xy);
    
    // // Prepare 4-component input vector:
    // // 1. sin(angular) component from cylindrical coordinates
    // // 2. vertical component from cylindrical coordinates  
    // // 3. sin(polar angle) of inverted ray in normal frame
    // // 4. sin(azimuthal angle) of inverted ray in normal frame
    // let cc = float4(
    //     sin(c.y),    // sin(theta) from cylindrical coordinates
    //     c.z,         // vertical component (normalized height)
    //     sinTheta,    // sin(polar angle) of inverted ray
    //     sinPhi       // sin(azimuthal angle) of inverted ray
    // );

    // return HSV(0.0, 1.0).eval(0.5 * cc.w + 0.5);

    float[7] cc = float[7](
        sin(c.y),
        cos(c.y),
        c.z,
        sinTheta,
        cosTheta,
        sinPhi,
        cosPhi,
    );
    
    return Network.eval(
        parameters,
        configuration,
        Network.InputVec(cc)
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