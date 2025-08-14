import camera;
import cmap;
import mesh;
import random;
import sdf;
import shapes;

import neural;
import network_with_addresses;

// Rendering the reference mesh
uniform RayFrame rayFrame;
uniform RWTexture2D targetTexture;
uniform uint2 targetResolution;

// Rendering the neural SDF
extern static const int Hidden;
extern static const int HiddenLayers;
extern static const int Levels;
extern struct ColorMap : IColorMap = Plasma;

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
        return forward(Network.InputVec.from(p))[0];
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void render_neural(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = (tid.xy + 0.5) / targetResolution;
    uv.y = 1 - uv.y;

    let ray = rayFrame.rayAt(uv);

    let intx = sphereTrace((Network) network, ray, 100, 1e-3, 10.0);
    // TODO: uniform to toggle between color map and normal
    float3 color = ColorMap(0, 100).eval(intx.steps);
    // float3 color = (intx.t > 0.0) ? intx.n * 0.5 + 0.5 : float3(0, 0, 0);
    targetTexture.Store(tid.xy, float4(color, 1));
}