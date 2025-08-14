import neural;

extern static const int In;

typealias Vec = InlineVector<float, In>;

// Vector arithmetic derivatives for In-element vectors
struct VectorArithmeticDerivativesGlobals
{
    RWStructuredBuffer<Vec> input_a;
    RWStructuredBuffer<Vec> input_b;
    RWStructuredBuffer<Vec> output_add_a;
    RWStructuredBuffer<Vec> output_add_b;
    RWStructuredBuffer<Vec> output_sub_a;
    RWStructuredBuffer<Vec> output_sub_b;
    RWStructuredBuffer<Vec> output_mul_a;
    RWStructuredBuffer<Vec> output_mul_b;
    RWStructuredBuffer<Vec> output_div_a;
    RWStructuredBuffer<Vec> output_div_b;
    uint count;
}

ParameterBlock<VectorArithmeticDerivativesGlobals> globals;

[BackwardDifferentiable]
Vec test_add(Vec a, Vec b)
{
    return a.add(b);
}

[BackwardDifferentiable]
Vec test_sub(Vec a, Vec b)
{
    return a.sub(b);
}

[BackwardDifferentiable]
Vec test_mul(Vec a, Vec b)
{
    return a.mul(b);
}

[BackwardDifferentiable]
Vec test_div(Vec a, Vec b)
{
    return a.div(b);
}

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    
    var a = globals.input_a[tid];
    var b = globals.input_b[tid];
    
    // Test addition derivatives
    var da_add = diffPair(a, Vec(0.0));
    var db_add = diffPair(b, Vec(0.0));
    bwd_diff(test_add)(da_add, db_add, Vec(1.0));
    globals.output_add_a[tid] = da_add.d;
    globals.output_add_b[tid] = db_add.d;
    
    // Test subtraction derivatives
    var da_sub = diffPair(a, Vec(0.0));
    var db_sub = diffPair(b, Vec(0.0));
    bwd_diff(test_sub)(da_sub, db_sub, Vec(1.0));
    globals.output_sub_a[tid] = da_sub.d;
    globals.output_sub_b[tid] = db_sub.d;
    
    // Test multiplication derivatives
    var da_mul = diffPair(a, Vec(0.0));
    var db_mul = diffPair(b, Vec(0.0));
    bwd_diff(test_mul)(da_mul, db_mul, Vec(1.0));
    globals.output_mul_a[tid] = da_mul.d;
    globals.output_mul_b[tid] = db_mul.d;
    
    // Test division derivatives
    var da_div = diffPair(a, Vec(0.0));
    var db_div = diffPair(b, Vec(0.0));
    bwd_diff(test_div)(da_div, db_div, Vec(1.0));
    globals.output_div_a[tid] = da_div.d;
    globals.output_div_b[tid] = db_div.d;
}