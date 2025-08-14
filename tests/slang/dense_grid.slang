import neural;

extern static const int Dim;
extern static const int Levels;
extern static const int Features;

typealias InputVec = InlineVector<float, Dim>;
typealias OutputVec = InlineVector<float, Features * Levels>;

// Dense grid encoding in space
typealias Storage = BindlessBufferStorage<float>;
typealias Encoder = DenseGridEncoder<float, Dim, Levels, Features, Storage>;

struct DenseGridParameters
{
    RWSampler1DArray<Storage, Levels> parameters;
    Array<int, Levels> resolutions;

    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<OutputVec> output;
};

ParameterBlock<DenseGridParameters> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    let encoder = Encoder(globals.parameters, globals.resolutions);
    let input = globals.input[tid];
    let output = encoder.eval<InputVec, OutputVec>(input);
    globals.output[tid] = output;
}