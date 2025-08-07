import neural;

typealias Vec3 = InlineVector<float, 3>;
typealias Vec24 = InlineVector<float, 24>;

struct FrequencyEncoderDerivativeGlobals
{
    RWStructuredBuffer<Vec3> input;
    RWStructuredBuffer<Vec3> output;
}

ParameterBlock<FrequencyEncoderDerivativeGlobals> globals;

[BackwardDifferentiable]
Vec24 frequency_encoder(Vec3 input)
{
    var encoder = FrequencyEncoder<float, 3, 4>();
    return encoder.eval<Vec3, Vec24>(input);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var input = globals.input[tid];
    var dinput = diffPair(input);
    bwd_diff(frequency_encoder)(dinput, Vec24(1.0));
    globals.output[tid] = dinput.d;
}