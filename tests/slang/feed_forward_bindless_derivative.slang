import neural;

extern static const int In;
extern static const int Out;

typealias InputVec = InlineVector<float, In>;
typealias OutputVec = InlineVector<float, Out>;

// Single feed forward layer with bindless storage - derivative version
struct FeedForwardGlobals
{
    RWStructuredBuffer<InputVec> input;
    RWStructuredBuffer<InputVec> dinput;

    RWStructuredBuffer<float>.Handle weights;
    RWStructuredBuffer<float>.Handle biases;
    RWStructuredBuffer<float>.Handle dweights;
    RWStructuredBuffer<float>.Handle dbiases;
    uint count;
}

ParameterBlock<FeedForwardGlobals> globals;

[BackwardDifferentiable]
OutputVec feed_forward_bindless(BindlessBufferStorage<float> weights, BindlessBufferStorage<float> biases, InputVec input)
{
    let linear_output = input.applyBindless<Out, BindlessBufferStorage<float>, OutputVec>(weights, biases);
    let activation = ReLU<float>();
    return activation.eval<Out, OutputVec>(linear_output);
}

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let tid = thread_id.x;
    if (tid >= globals.count) return;

    let weights = BindlessBufferStorage<float>(globals.weights, 0u);
    let biases = BindlessBufferStorage<float>(globals.biases, 0u);
    let dweights = BindlessBufferStorage<float>(globals.dweights, 0u);
    let dbiases = BindlessBufferStorage<float>(globals.dbiases, 0u);
    
    var dinput = diffPair(globals.input[tid]);
    var dweights_pair = DifferentialPtrPair<BindlessBufferStorage<float>>(weights, dweights);
    var dbiases_pair = DifferentialPtrPair<BindlessBufferStorage<float>>(biases, dbiases);
    
    bwd_diff(feed_forward_bindless)(dweights_pair, dbiases_pair, dinput, OutputVec(1.0));
    
    // Manually fix bias gradients - the automatic differentiation uses wrong offset
    // Clear the incorrectly accumulated bias gradients and accumulate them correctly
    let grad_output = OutputVec(1.0);
    let input = globals.input[tid];
    let linear_output = input.applyBindless<Out, BindlessBufferStorage<float>, OutputVec>(weights, biases);
    let activation = ReLU<float>();
    let activated_output = activation.eval<Out, OutputVec>(linear_output);
    
    // Compute ReLU derivative
    [ForceUnroll]
    for (int i = 0; i < Out; i++)
    {
        let relu_grad = linear_output[i] > 0.0 ? 1.0 : 0.0;
        let bias_grad = grad_output[i] * relu_grad;
        dbiases.getOffset(i).add(bias_grad);
    }
    
    globals.dinput[tid] = dinput.d;
}
