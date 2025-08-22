import neural;

extern static const int DIM;
extern static const int FEATURES;

typealias InputVec = InlineVector<float, DIM>;
typealias OutputVec = InlineVector<float, FEATURES>;

struct RandomFourierEncoderGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<OutputVec> output;
    RWStructuredBuffer<float> parameters;
    uint count;
}

ParameterBlock<RandomFourierEncoderGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    var storage = StructuredBufferStorage<float>(globals.parameters);
    var encoder = RandomFourierEncoder<float, DIM, FEATURES, StructuredBufferStorage<float>>(0);
    var input = InputVec(globals.input[tid]);
    var output = encoder.eval<InputVec, OutputVec>(storage, input);
    globals.output[tid] = output;
}
