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

// ReLU derivative for vector of 2 elements
struct VectorReLUDerivativeGlobals
{
    RWStructuredBuffer<float2> input;
    RWStructuredBuffer<float2> output;
}

ParameterBlock<VectorReLUDerivativeGlobals> vector_relu_derivative_globals;

[BackwardDifferentiable]
InlineVector<float, 2> eval(no_diff ReLU<float> activation, InlineVector<float, 2> input)
{
    return activation.eval(input);
}

[shader("compute")]
void vector_relu_derivative_main(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var activation = ReLU<float>();
    var input = InlineVector<float, 2>(vector_relu_derivative_globals.input[tid]);
    var dinput = diffPair(input, InlineVector<float, 2>(0.0));
    bwd_diff(eval)(activation, dinput, InlineVector<float, 2>(1.0));
    vector_relu_derivative_globals.output[tid] = float2(dinput.d[0], dinput.d[1]);
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

// // Single feed forward layer
// typealias Vec8 = InlineVector<float, 8>;

// struct FeedForwardGlobals
// {
//     RWStructuredBuffer<Vec16> input;
//     RWStructuredBuffer<Vec8> output;
    
//     RWStructuredBuffer<float> parameters;
// }

// ParameterBlock<FeedForwardGlobals> feed_forward_globals;

// [shader("compute")]
// void feed_forward_main(uint3 thread_id : SV_DispatchThreadID)
// {
//     let parameters = StructuredBufferStorage<float>(feed_forward_globals.parameters);
//     let tid = thread_id.x;
//     let network = FeedForward<float, 16, 8, StructuredBufferStorage<float>, ReLU<float>>(0, {});
//     let input = Vec16(feed_forward_globals.input[tid]);
//     let output = network.eval<Vec16, Vec8>(parameters, input);
//     feed_forward_globals.output[tid] = output;
// }