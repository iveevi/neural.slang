import neural;

typealias Vec4 = InlineVector<float, 4>;

// Vector arithmetic derivatives for 4-element vectors
struct VectorArithmeticDerivativesGlobals
{
    RWStructuredBuffer<Vec4> input_a;
    RWStructuredBuffer<Vec4> input_b;
    RWStructuredBuffer<Vec4> output_add_a;
    RWStructuredBuffer<Vec4> output_add_b;
    RWStructuredBuffer<Vec4> output_sub_a;
    RWStructuredBuffer<Vec4> output_sub_b;
    RWStructuredBuffer<Vec4> output_mul_a;
    RWStructuredBuffer<Vec4> output_mul_b;
    RWStructuredBuffer<Vec4> output_div_a;
    RWStructuredBuffer<Vec4> output_div_b;
}

ParameterBlock<VectorArithmeticDerivativesGlobals> globals;

[BackwardDifferentiable]
Vec4 test_add(Vec4 a, Vec4 b)
{
    return a.add(b);
}

[BackwardDifferentiable]
Vec4 test_sub(Vec4 a, Vec4 b)
{
    return a.sub(b);
}

[BackwardDifferentiable]
Vec4 test_mul(Vec4 a, Vec4 b)
{
    return a.mul(b);
}

[BackwardDifferentiable]
Vec4 test_div(Vec4 a, Vec4 b)
{
    return a.div(b);
}

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    
    var a = globals.input_a[tid];
    var b = globals.input_b[tid];
    
    // Test addition derivatives
    var da_add = diffPair(a, Vec4(0.0));
    var db_add = diffPair(b, Vec4(0.0));
    bwd_diff(test_add)(da_add, db_add, Vec4(1.0));
    globals.output_add_a[tid] = da_add.d;
    globals.output_add_b[tid] = db_add.d;
    
    // Test subtraction derivatives
    var da_sub = diffPair(a, Vec4(0.0));
    var db_sub = diffPair(b, Vec4(0.0));
    bwd_diff(test_sub)(da_sub, db_sub, Vec4(1.0));
    globals.output_sub_a[tid] = da_sub.d;
    globals.output_sub_b[tid] = db_sub.d;
    
    // Test multiplication derivatives
    var da_mul = diffPair(a, Vec4(0.0));
    var db_mul = diffPair(b, Vec4(0.0));
    bwd_diff(test_mul)(da_mul, db_mul, Vec4(1.0));
    globals.output_mul_a[tid] = da_mul.d;
    globals.output_mul_b[tid] = db_mul.d;
    
    // Test division derivatives
    var da_div = diffPair(a, Vec4(0.0));
    var db_div = diffPair(b, Vec4(0.0));
    bwd_diff(test_div)(da_div, db_div, Vec4(1.0));
    globals.output_div_a[tid] = da_div.d;
    globals.output_div_b[tid] = db_div.d;
}