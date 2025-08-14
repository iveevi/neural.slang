import neural;

typealias Optimizer = SGD<float>;
typealias OptimizerState = Optimizer.State;

struct SGDGlobals
{
    RWStructuredBuffer<OptimizerState> state;
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> gradients;
    float lr;
    float momentum;
    uint count;
}

ParameterBlock<SGDGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    let optimizer = Optimizer(globals.lr, globals.momentum);
    optimizer.step(globals.state[tid], globals.parameters[tid], globals.gradients[tid]);
}
