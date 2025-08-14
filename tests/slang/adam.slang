import neural;

typealias Optimizer = Adam<float>;
typealias OptimizerState = Optimizer.State;

struct AdamGlobals
{
    RWStructuredBuffer<OptimizerState> state;
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> gradients;
    uint count;
}

ParameterBlock<AdamGlobals> globals;

// TODO: test different lr/betas/eps with uniforms...
[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    let optimizer = Optimizer();
    optimizer.step(globals.state[tid], globals.parameters[tid], globals.gradients[tid]);
}