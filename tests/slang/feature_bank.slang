import neural;

extern static const int Features;

typealias OutputVec = InlineVector<float, Features>;
typealias Storage = StructuredBufferStorage<float>;

struct FeatureBankGlobals
{
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<uint> indices;
    RWStructuredBuffer<OutputVec> output;
    uint count;
}

ParameterBlock<FeatureBankGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    let storage = Storage(globals.parameters);
    let bank = FeatureBank<float, Features, Storage>(0);
    var index = int(globals.indices[tid]);
    var output = bank.eval<OutputVec>(storage, index);
    globals.output[tid] = output;
}
