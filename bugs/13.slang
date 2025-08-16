// Crahes because of the IDifferentiable constraint

interface IVector<T, int N> : IDifferentiable
    where T : __BuiltinFloatingPointType
{
    __init();
    
    // Also crahses with this:
    // [BackwardDifferentiable]
    OutputVector apply<int OutputSize, OutputVector>(
        RWStructuredBuffer<T> buffer,
        uint address
    )
        where OutputVector : IVector<T, OutputSize>;
}

struct InlineVector<T, int N> : IVector<T, N>
    where T : __BuiltinFloatingPointType
{
    typealias Differential = This;
    
    __init() {}
    
    // Also crahses with this:
    // [BackwardDifferentiable]
    OutputVector apply<int OutputSize, OutputVector>(
        RWStructuredBuffer<T> buffer,
        uint address
    )
        where OutputVector : IVector<T, OutputSize>
    {
        return OutputVector();
    }
}

uniform RWStructuredBuffer<float> buffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    // This line triggers the IR validation error
    let result = InlineVector<float, 1>().apply<1, InlineVector<float, 1>>(buffer, 0);
}