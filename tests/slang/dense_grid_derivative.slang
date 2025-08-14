import neural;

extern static const int Dim;
extern static const int Levels;
extern static const int Features;

typealias InputVec = InlineVector<float, Dim>;
typealias OutputVec = InlineVector<float, Features * Levels>;

// Dense grid encoding in space
typealias Storage = StructuredBufferStorage<float>;
typealias Encoder = DenseGridEncoder<float, Dim, Levels, Features, Storage>;

struct DenseGridParameters
{
    Storage parameters;
    Array<int, Levels> resolutions;

    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<InputVec> dinput;
};

ParameterBlock<DenseGridParameters> parameters;

[BackwardDifferentiable]
OutputVec eval(Array<Storage, Levels> addresses, InputVec input)
{
    var encoder = Encoder(addresses, parameters.resolutions);
    return encoder.eval<InputVec, OutputVec>(input);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    var inputDiff = diffPair(parameters.input[tid]);
    var addressesDiff = DifferentialPtrPair<Array<Storage, Levels>>(parameters.addresses, parameters.daddresses);
    bwd_diff(eval)(addressesDiff, inputDiff, OutputVec(1.0));
    parameters.dinput[tid] = inputDiff.d;
}