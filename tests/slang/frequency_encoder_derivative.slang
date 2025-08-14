import neural;

extern static const int In;
extern static const int Levels;

typealias InputVec = InlineVector<float, In>;
typealias OutputVec = InlineVector<float, 2 * Levels * In>;

struct FrequencyEncoderDerivativeGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<InputVec> dinput;
    uint count;
}

ParameterBlock<FrequencyEncoderDerivativeGlobals> globals;

[BackwardDifferentiable]
OutputVec frequency_encoder(InputVec input)
{
    var encoder = FrequencyEncoder<float, In, Levels>();
    return encoder.eval<InputVec, OutputVec>(input);
}

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    var input = globals.input[tid];
    var dinput = diffPair(input);
    bwd_diff(frequency_encoder)(dinput, OutputVec(1.0));
    globals.dinput[tid] = dinput.d;
}