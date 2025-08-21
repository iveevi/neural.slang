import neural;
import ngp;

extern static const int Hidden = 64;
extern static const int HiddenLayers = 2;
extern static const int Levels = 8;

typealias MLP = AddressBasedMLP<Encoder.Out, 1, Hidden, HiddenLayers, ReLU<float>, Identity<float>>;
typealias Encoder = FrequencyEncoder<float, 1, Levels>;

uniform MLP mlp;
uniform Encoder encoder;

uniform RWStructuredBuffer<float> inputBuffer;
uniform RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
float forward(
    MLP.Parameters parameters,
    no_diff float input)
{
    let inputVec = InlineVector<float, 1>.fromVector(input);
    let encoded = Encoder().eval<
        InlineVector<float, 1>,
        InlineVector<float, Encoder.Out>
    >(inputVec);
    return mlp.eval(mlp.parameters, encoded).toVector()[0];
}

[shader("compute")]
[numthreads(32, 1, 1)]
void forward(uint3 tid : SV_DispatchThreadID)
{
    let input = inputBuffer[tid.x];
    outputBuffer[tid.x] = forward(mlp.parameters, input);
}

// Backward pass
uniform RWStructuredBuffer<float> expectedBuffer;
uniform float boost;

[BackwardDifferentiable]
float loss(
    MLP.Parameters parameters,
    no_diff float input,
    no_diff float expected)
{
    let inputVec = InlineVector<float, 1>.fromVector(input);
    let encoded = Encoder().eval<
        InlineVector<float, 1>,
        InlineVector<float, Encoder.Out>
    >(inputVec);
    let predicted = mlp.eval(parameters, encoded);
    let expectedVec = InlineVector<float, 1>.fromVector(expected);
    return MSE<float>.eval(predicted, expectedVec);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void backward(uint3 tid : SV_DispatchThreadID)
{
    bwd_diff(loss)(
        mlp.dual,
        inputBuffer[tid.x],
        expectedBuffer[tid.x],
        boost
    );
}
