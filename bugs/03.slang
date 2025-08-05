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

    [mutating]
    void write(int index, T value);
    
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

    [mutating]
    public void write(int index, T value)
    {
        data[index] = value;
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

public struct ReLU<T> where T : __BuiltinFloatingPointType
{
    [NoDiffThis, BackwardDerivative(evalBwd)]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential : IVector<T, N>
    {
        return input.max(T(0.0));
    }

    void evalBwd<int N, Vector>(inout DifferentialPair<Vector> dinput, Vector.Differential doutput)
        where Vector : IVector<T, N>
        where Vector.Differential : IVector<T, N>
    {
        var d = Vector.Differential(doutput);
        for (int i = 0; i < N; i++)
        {
            if (dinput.p[i] < T(0.0))
                d[i] = T(0.0);
        }

        dinput = DifferentialPair<Vector>(dinput.p, d);
    }
}

RWStructuredBuffer<float> g_output;

[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    var input = InlineVector<float, 2>(0.0);
    var output = ReLU<float>().eval<2, InlineVector<float, 2>>(input);
    g_output[thread_id.x] = output[0];
}