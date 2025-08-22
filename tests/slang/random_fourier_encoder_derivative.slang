import neural;

extern static const int DIM;
extern static const int FEATURES;

typealias InputVec = InlineVector<float, DIM>;
typealias OutputVec = InlineVector<float, FEATURES>;

struct RandomFourierEncoderDerivativeGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<InputVec> dinput;
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> dparameters;
    uint count;
}

ParameterBlock<RandomFourierEncoderDerivativeGlobals> globals;

[BackwardDifferentiable]
OutputVec random_fourier_encoder(InputVec input, StructuredBufferStorage<float> storage)
{
    var encoder = RandomFourierEncoder<float, DIM, FEATURES, StructuredBufferStorage<float>>(0);
    return encoder.eval<InputVec, OutputVec>(storage, input);
}

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    var input = globals.input[tid];
    var storage = StructuredBufferStorage<float>(globals.parameters);
    var dstorage = StructuredBufferStorage<float>(globals.dparameters);
    
    var dinput = diffPair(input);
    var dstorage_pair = DifferentialPtrPair<StructuredBufferStorage<float>>(storage, dstorage);
    
    bwd_diff(random_fourier_encoder)(dinput, dstorage_pair, OutputVec(1.0));
    
    globals.dinput[tid] = dinput.d;
}
