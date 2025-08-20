import neural;
import ngp;

extern static const int Hidden = 32;
extern static const int HiddenLayers = 2;
extern static const int Levels = 8;

typealias MLP = AddressBasedMLP<Encoder.Out, 3, Hidden, HiddenLayers, ReLU<float>, Identity<float>>;
typealias Encoder = FrequencyEncoder<float, 2, Levels>;

uniform MLP mlp;
uniform Encoder encoder;

uniform RWTexture2D targetTexture;
uniform uint2 targetResolution;

[BackwardDifferentiable]
float3 forward(
    MLP.Parameters parameters,
    no_diff float2 uv)
{
    let input = InlineVector<float, 2>.fromVector(uv);
    let encoded = Encoder().eval<
        InlineVector<float, 2>,
        InlineVector<float, Encoder.Out>
    >(input);
    return mlp.eval(mlp.parameters, encoded).toVector();
}

[shader("compute")]
[numthreads(32, 1, 1)]
void render_neural(uint3 tid : SV_DispatchThreadID)
{
    let uv = (tid.xy + 0.5) / targetResolution;
    targetTexture[tid.xy] = float4(forward(mlp.parameters, uv), 1.0);
}

// Backward pass
uniform RWStructuredBuffer<float2> samples;
uniform RWStructuredBuffer<float3, ScalarDataLayout> expected;
uniform float boost;

[BackwardDifferentiable]
float loss(
    MLP.Parameters parameters,
    no_diff float2 uv,
    no_diff float3 expected)
{
    // TODO: overload for vector<T, N>...
    let input = InlineVector<float, 2>.fromVector(uv);
    let encoded = Encoder().eval<
        InlineVector<float, 2>,
        InlineVector<float, Encoder.Out>
    >(input);
    let predicted = mlp.eval(parameters, encoded);
    let expectedVector = InlineVector<float, 3>.fromVector(expected);
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
        samples[tid.x],
        expected[tid.x],
        boost
    );
}