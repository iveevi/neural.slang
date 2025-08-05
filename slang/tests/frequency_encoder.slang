import neural;

typealias Vec3 = InlineVector<float, 3>;
typealias Vec24 = InlineVector<float, 24>;

struct FrequencyEncoderGlobals
{
    RWStructuredBuffer<Vec3> input;
    RWStructuredBuffer<Vec24> output;
}

ParameterBlock<FrequencyEncoderGlobals> frequency_encoder_globals;

[shader("compute")]
void frequency_encoder_main(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var encoder = FrequencyEncoder<float, 3, 4>();
    var input = Vec3(frequency_encoder_globals.input[tid]);
    var output = encoder.eval<Vec3, Vec24>(input);
    frequency_encoder_globals.output[tid] = output;
}