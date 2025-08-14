import neural;

extern static const int Features;

typealias OutputVec = InlineVector<float, Features>;
typealias Storage = StructuredBufferStorage<float>;

struct FeatureBankDerivativeGlobals
{
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> dparameters;
    RWStructuredBuffer<uint> indices;
    RWStructuredBuffer<OutputVec> output;
    uint count;
}

ParameterBlock<FeatureBankDerivativeGlobals> globals;

[BackwardDifferentiable]
OutputVec feature_bank_eval(Storage storage, int index)
{
    let bank = FeatureBank<float, Features, Storage>(0);
    return bank.eval<OutputVec>(storage, index);
}

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    let index = int(globals.indices[tid]);
    var storage = Storage(globals.parameters);
    var dstorage = Storage(globals.dparameters);
    var storageDiff = DifferentialPtrPair<Storage>(storage, dstorage);
    
    bwd_diff(feature_bank_eval)(storageDiff, index, OutputVec(1.0));

    globals.output[tid] = OutputVec(1.0);
}
