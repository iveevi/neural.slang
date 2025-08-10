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
}

ParameterBlock<SGDGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    let optimizer = Optimizer(globals.lr, globals.momentum);

    int tid = thread_id.x;
    optimizer.step(globals.state[tid], globals.parameters[tid], globals.gradients[tid]);
}
