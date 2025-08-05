interface IVector<T, int N> : IDifferentiable, IArithmetic
    where T : __BuiltinFloatingPointType {}

struct Vector<T, int N>
    where T : __BuiltinFloatingPointType
{
    T data[N];

    __init() {}
    __init(int value) {}
    __init(This other) {}

    __subscript(uint index) -> T
    {
        get { return data[index]; }
        set { data[index] = newValue; }
    }

    [Differentiable]
    This add(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] += this[i];

        return this;
    }

    This sub(This other) {}
    This mul(This other) {}
    This div(This other) {}
    This mod(This other) {}
    This neg() {}

    bool equals(This other) {}
    bool lessThan(This other) {}
    bool lessThanOrEquals(This other) {}
}

extension<T, int N> Vector<T, N> : IVector<T, N>
    where T : __BuiltinFloatingPointType {}