import neural;

// ReLU for scalars
struct ReLUGlobals
{
    RWStructuredBuffer<float> input;
    RWStructuredBuffer<float> output;
}

ParameterBlock<ReLUGlobals> relu_globals;

[shader("compute")]
void relu_main(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = ReLU<float>();
    var input = InlineVector<float, 1>(relu_globals.input[tid]);
    var output = activation.eval(input);
    relu_globals.output[tid] = output[0];
}

// ReLU for vector of 2 elements
struct VectorReLUGlobals
{
    RWStructuredBuffer<float2> input;
    RWStructuredBuffer<float2> output;
}

ParameterBlock<VectorReLUGlobals> vector_relu_globals;

[shader("compute")]
void vector_relu_main(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = ReLU<float>();
    var input = InlineVector<float, 2>(vector_relu_globals.input[tid]);
    var output = activation.eval(input);
    vector_relu_globals.output[tid] = float2(output[0], output[1]);
}

// Mean squared error for vector of 16 elements
typealias Vec16 = InlineVector<float, 16>;

struct MSEGlobals
{
    RWStructuredBuffer<Vec16> input;
    RWStructuredBuffer<Vec16> target;
    RWStructuredBuffer<float> output;
}

ParameterBlock<MSEGlobals> mse_globals;

static MeanSquaredError mse = MeanSquaredError();

[shader("compute")]
void mse_main(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var input = mse_globals.input[tid];
    var target = mse_globals.target[tid];
    var loss = mse.eval(input, target);
    mse_globals.output[tid] = loss;
}