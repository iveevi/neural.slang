import neural;

typealias Vec3 = InlineVector<float, 3>;
typealias Vec24 = InlineVector<float, 24>;

struct FrequencyEncoderGlobals
{
    RWStructuredBuffer<Vec3> input;
    RWStructuredBuffer<Vec24> output;
}

ParameterBlock<FrequencyEncoderGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var encoder = FrequencyEncoder<float, 3, 4>();
    var input = Vec3(globals.input[tid]);
    var output = encoder.eval<Vec3, Vec24>(input);
    globals.output[tid] = output;
}