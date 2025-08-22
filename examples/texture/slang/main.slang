import neural;
import ngp;

extern static const int Hidden = 32;
extern static const int HiddenLayers = 2;
extern static const int Levels = 8;

typealias Encoder = RandomFourierFeatures<2, 32>;
typealias MLP = AddressBasedMLP<Encoder.Out, 3, Hidden, HiddenLayers, ReLU<float>, Identity<float>>;

uniform Encoder encoder;
uniform MLP mlp;

uniform RWTexture2D targetTexture;
uniform uint2 targetResolution;

[BackwardDifferentiable]
float3 forward(
    MLP.Parameters mlpParameters,
    Encoder.Parameters encoderParameters,
    no_diff float2 uv)
{
    let input = InlineVector<float, 2>.fromVector(uv);
    let encoded = encoder.eval(encoderParameters, input);
    return mlp.eval(mlpParameters, encoded).toVector();
}

[shader("compute")]
[numthreads(32, 1, 1)]
void render_neural(uint3 tid : SV_DispatchThreadID)
{
    let uv = (tid.xy + 0.5) / targetResolution;
    targetTexture[tid.xy] = float4(forward(mlp.parameters, encoder.parameters, uv), 1.0);
}

// Backward pass
uniform RWStructuredBuffer<float2> samples;
uniform RWStructuredBuffer<float3, ScalarDataLayout> expected;
uniform float boost;

[BackwardDifferentiable]
float loss(
    MLP.Parameters mlpParameters,
    Encoder.Parameters encoderParameters,
    no_diff float2 uv,
    no_diff float3 expected)
{
    let input = InlineVector<float, 2>.fromVector(uv);
    let encoded = encoder.eval(encoderParameters, input);
    let predicted = mlp.eval(mlpParameters, encoded);
    let expectedVector = InlineVector<float, 3>.fromVector(expected);
    // TODO: loss overload for vector<T, N>...
    return MSE<float>.eval(
        predicted,
        expectedVector
    );
}

[shader("compute")]
[numthreads(32, 1, 1)]
void backward(uint3 tid : SV_DispatchThreadID)
{
    bwd_diff(loss)(
        mlp.dual,
        encoder.dual,
        samples[tid.x],
        expected[tid.x],
        boost
    );
}