import neural;
import network_with_addresses;

extern static const int In;
extern static const int Out;
extern static const int Hidden;
extern static const int HiddenLayers;
extern static const int Levels;

typealias Encoder = FrequencyEncoder<float, In, Levels>;
typealias Network = NetworkWithAddresses<
    In,
    Out,
    Encoder.Out,
    Hidden,
    HiddenLayers,
    Encoder,
    ReLU<float>
>;

typealias InputVec = Network.InputVec;
typealias OutputVec = Network.OutputVec;

ParameterBlock<Network> network;

struct ForwardInputs
{
    RWStructuredBuffer<InputVec> inputBuffer;
    RWStructuredBuffer<OutputVec> outputBuffer;
}

ParameterBlock<ForwardInputs> forwardInputs;

// Neural network inference
[shader("compute")]
[numthreads(32, 1, 1)]
void forward(uint3 thread_id : SV_DispatchThreadID)
{
    // TODO: pad up to 64 threads for saturated work groups
    int tid = thread_id.x;

    let input = InputVec(forwardInputs.inputBuffer[tid]);
    let output = network.forward(input);
    forwardInputs.outputBuffer[tid] = output;
}

// Neural network training
struct BackwardInputs
{
    RWStructuredBuffer<InputVec> inputBuffer;
    RWStructuredBuffer<OutputVec> expectedBuffer;
    float boost;
}

ParameterBlock<BackwardInputs> backwardInputs;

[shader("compute")]
[numthreads(32, 1, 1)]
void backward(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;

    let input = InputVec(backwardInputs.inputBuffer[tid]);
    let expected = OutputVec(backwardInputs.expectedBuffer[tid]);

    network.backward(input, expected, backwardInputs.boost);
}

// Neural network optimization
[shader("compute")]
[numthreads(32, 1, 1)]
void optimize(uint3 thread_id : SV_DispatchThreadID)
{
    network.optimize(thread_id.x);
}
