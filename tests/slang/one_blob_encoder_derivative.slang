import neural;

extern static const int N;
extern static const int K;

typealias InputVec = InlineVector<float, N>;
typealias OutputVec = InlineVector<float, N * K>;

struct OneBlobEncoderDerivativeGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<InputVec> dinput;
    uint count;
}

ParameterBlock<OneBlobEncoderDerivativeGlobals> globals;

[BackwardDifferentiable]
OutputVec one_blob_encoder(InputVec input)
{
    var encoder = OneBlobEncoder<float, N, K>();
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
    bwd_diff(one_blob_encoder)(dinput, OutputVec(1.0));
    globals.dinput[tid] = dinput.d;
}
