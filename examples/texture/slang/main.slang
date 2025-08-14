import neural;
import network_with_addresses;

extern static const int Hidden;
extern static const int HiddenLayers;
extern static const int Levels;

typealias Encoder = FrequencyEncoder<float, 2, Levels>;

typealias Network = NetworkWithAddresses<
    2, 3, Encoder.Out,
    Hidden, HiddenLayers,
    Encoder,
    ReLU<float>,
>;

ParameterBlock<Network> network;

uniform RWTexture2D targetTexture;
uniform uint2 targetResolution;

[shader("compute")]
[numthreads(32, 1, 1)]
void render_neural(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = (tid.xy + 0.5) / targetResolution;
    float3 color = network.forward(Network.InputVec.fromVector(uv)).toVector();
    targetTexture[tid.xy] = float4(color, 1.0);
}