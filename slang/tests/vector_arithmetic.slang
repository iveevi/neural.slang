import neural;

typealias Vec4 = InlineVector<float, 4>;

// Vector arithmetic operations for 4-element vectors
struct VectorArithmeticGlobals
{
    RWStructuredBuffer<Vec4> input_a;
    RWStructuredBuffer<Vec4> input_b;
    RWStructuredBuffer<Vec4> output_add;
    RWStructuredBuffer<Vec4> output_sub;
    RWStructuredBuffer<Vec4> output_mul;
    RWStructuredBuffer<Vec4> output_div;
}

ParameterBlock<VectorArithmeticGlobals> globals;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    
    var a = globals.input_a[tid];
    var b = globals.input_b[tid];
    
    // Test all arithmetic operations
    globals.output_add[tid] = a.add(b);
    globals.output_sub[tid] = a.sub(b);
    globals.output_mul[tid] = a.mul(b);
    globals.output_div[tid] = a.div(b);
}