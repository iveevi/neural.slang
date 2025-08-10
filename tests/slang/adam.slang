import neural;

typealias Optimizer = Adam<float>;
typealias OptimizerState = Optimizer.State;

struct AdamGlobals
{
    RWStructuredBuffer<OptimizerState> state;
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> gradients;
}

ParameterBlock<AdamGlobals> globals;

// TODO: test different lr/betas/eps with uniforms...
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let optimizer = Optimizer();

    int tid = thread_id.x;
    optimizer.step(globals.state[tid], globals.parameters[tid], globals.gradients[tid]);
}