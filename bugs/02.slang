interface IVector<T, int N> : IDifferentiable
    where T : __BuiltinFloatingPointType
{
    __init();
    __init(T value);

    __subscript(int index) -> T
    {
        get;
        set;
    }
    
    // Specific methods required for ML operations
    [BackwardDifferentiable] This max(T other);
    [BackwardDifferentiable] This max(This other);
}

public struct InlineVector<T, int N> : IVector<T, N>
    where T : __BuiltinFloatingPointType
{
    public typealias Differential = This;

    internal T[N] data;
    
    public __init() {}
    public __init(int value) { data[0] = T(value); }
    public __init(T value) { data[0] = value; }
    public __init(This other) { this.data = other.data; }

    public __subscript(int index) -> T
    {
        [BackwardDifferentiable]
        get() { return data[index]; }

        set() { data[index] = newValue; }
    }

    // Additional ML operations
    [BackwardDifferentiable]
    public This max(T other)
    {
        This result = this;

        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            result[i] = max(result[i], other);

        return result;
    }
    
    [BackwardDifferentiable]
    public This max(This other)
    {
        This result = this;

        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            result[i] = max(result[i], other[i]);

        return result;
    }
}

RWStructuredBuffer<float> g_output;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    var input = InlineVector<float, 2>(0.0);
    var output = input.max(1.0);
    g_output[thread_id.x] = output[0];
}