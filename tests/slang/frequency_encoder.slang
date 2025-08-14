import neural;

extern static const int In;
extern static const int Levels;

typealias InputVec = InlineVector<float, In>;
typealias OutputVec = InlineVector<float, 2 * Levels * In>;

struct FrequencyEncoderGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<OutputVec> output;
    uint count;
}

ParameterBlock<FrequencyEncoderGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    var encoder = FrequencyEncoder<float, In, Levels>();
    var input = InputVec(globals.input[tid]);
    var output = encoder.eval<InputVec, OutputVec>(input);
    globals.output[tid] = output;
}