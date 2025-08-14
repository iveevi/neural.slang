import activations;

extern struct Activation : IActivation<float>;
// extern struct Activation : IActivation;

uniform RWStructuredBuffer<Activation> activation;
uniform RWStructuredBuffer<float> input;
uniform RWStructuredBuffer<float> output;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    var x = input[tid];
    var y = activation[0].eval(x);
    output[tid] = y;
}