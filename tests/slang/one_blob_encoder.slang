import neural;

extern static const int N;
extern static const int K;

typealias InputVec = InlineVector<float, N>;
typealias OutputVec = InlineVector<float, N * K>;

struct OneBlobEncoderGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<OutputVec> output;
    uint count;
}

ParameterBlock<OneBlobEncoderGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    var encoder = OneBlobEncoder<float, N, K>();
    var input = InputVec(globals.input[tid]);
    var output = encoder.eval<InputVec, OutputVec>(input);
    globals.output[tid] = output;
}
