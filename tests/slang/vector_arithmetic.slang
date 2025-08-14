import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// Vector arithmetic operations for In-element vectors
struct VectorArithmeticGlobals
{
    RWStructuredBuffer<Vec> input_a;
    RWStructuredBuffer<Vec> input_b;
    RWStructuredBuffer<Vec> output_add;
    RWStructuredBuffer<Vec> output_sub;
    RWStructuredBuffer<Vec> output_mul;
    RWStructuredBuffer<Vec> output_div;
    uint count;
}

ParameterBlock<VectorArithmeticGlobals> globals;

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;
    
    var a = globals.input_a[tid];
    var b = globals.input_b[tid];
    
    // Test all arithmetic operations
    globals.output_add[tid] = a.add(b);
    globals.output_sub[tid] = a.sub(b);
    globals.output_mul[tid] = a.mul(b);
    globals.output_div[tid] = a.div(b);
}