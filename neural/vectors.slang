implementing neural;

////////////////////
// Inline vectors //
////////////////////

public struct InlineVector<T, int N> : IVector<T, N>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    public static const int Dimension = N;

    public typealias Differential = This;

    internal T[N] data;
    
    public __init() {
        for (int i = 0; i < N; i++)
            data[i] = T(0.0);
    }
    
    public __init(int value) {
        for (int i = 0; i < N; i++)
            data[i] = T(value);
    }
    
    public __init(T value) {
        for (int i = 0; i < N; i++)
            data[i] = value;
    }

    public __init(T[N] data) { this.data = data; }

    public __init(This other) { this.data = other.data; }

    public __subscript(int index) -> T
    {
        [BackwardDifferentiable]
        get() { return getIndex(index); }

        set() { data[index] = newValue; }
    }

    [BackwardDerivative(getIndexBwd)]
    public T getIndex(int index)
    {
        return data[index];
    }

    static void getIndexBwd(inout DifferentialPair<This> dthis, int index, T.Differential dloss)
    {
        var d = dthis.d;
        d.data[index] += __slang_noop_cast<T>(dloss);
        dthis = DifferentialPair<This>(dthis.p, d);
    }

    // Autodiff dX
    static Differential dzero()
    {
        return This(T(0.0));
    }

    static Differential dadd(Differential a, Differential b)
    {
        return a.add(b);
    }

    static Differential dmul(Differential a, Differential b)
    {
        return a.mul(b);
    }

    // Arithmetic operations
    [BackwardDerivative(addBwd)]
    public This add(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] = this[i] + other[i];

        return other;
    }

    static void addBwd(
        inout DifferentialPair<This> dthis,
        inout DifferentialPair<This> dother,
        Differential doutput
    )
    {
        dthis = DifferentialPair<This>(dthis.p, dadd(dthis.d, doutput));
        dother = DifferentialPair<This>(dother.p, dadd(dother.d, doutput));
    }
    
    [BackwardDerivative(subBwd)]
    public This sub(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] = this[i] - other[i];

        return other;
    }
    
    static void subBwd(
        inout DifferentialPair<This> dthis,
        inout DifferentialPair<This> dother,
        Differential doutput
    )
    {
        dthis = DifferentialPair<This>(dthis.p, dadd(dthis.d, doutput));
        dother = DifferentialPair<This>(dother.p, dadd(dother.d, doutput.neg()));
    }
    
    [BackwardDerivative(mulBwd)]
    public This mul(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] = this[i] * other[i];

        return other;
    }

    static void mulBwd(
        inout DifferentialPair<This> dthis,
        inout DifferentialPair<This> dother,
        Differential doutput
    )
    {
        dthis = DifferentialPair<This>(dthis.p, dadd(dthis.d, dmul(dother.p, doutput)));
        dother = DifferentialPair<This>(dother.p, dadd(dother.d, dmul(dthis.p, doutput)));
    }

    [BackwardDerivative(divBwd)]
    public This div(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] = this[i] / other[i];

        return other;
    }

    static void divBwd(
        inout DifferentialPair<This> dthis,
        inout DifferentialPair<This> dother,
        Differential doutput
    )
    {
        // For c = a / b: dc/da = 1/b * dc, dc/db = -a/(b^2) * dc
        let one_over_b = This(T(1.0)).div(dother.p);
        let neg_a_over_b_squared = dthis.p.neg().div(dother.p).div(dother.p);
        
        dthis = DifferentialPair<This>(dthis.p, dadd(dthis.d, dmul(one_over_b, doutput)));
        dother = DifferentialPair<This>(dother.p, dadd(dother.d, dmul(neg_a_over_b_squared, doutput)));
    }
    
    public This mod(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] = this[i] % other[i];

        return other;
    }

    public This neg()
    {
        This result = this;

        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            result[i] = -this[i];

        return result;
    }

    // Comparison operations
    public bool equals(This other)
    {
        for (int i = 0; i < N; i++)
            if (this.data[i] != other.data[i]) return false;
        return true;
    }
    
    public bool lessThan(This other)
    {
        for (int i = 0; i < N; i++)
            if (this.data[i] >= other.data[i]) return false;
        return true;
    }
    
    public bool lessThanOrEquals(This other)
    {
        for (int i = 0; i < N; i++)
            if (this.data[i] > other.data[i]) return false;
        return true;
    }
    
    // Additional ML operations
    public This max(T other)
    {
        This result = this;

        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            result.data[i] = max(result.data[i], other);

        return result;
    }
    
    public This max(This other)
    {
        This result = this;

        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            result.data[i] = max(result.data[i], other.data[i]);

        return result;
    }

    public This step(T threshold)
    {
        This result = this;

        for (int i = 0; i < N; i++)
            result.data[i] = (result.data[i] > threshold) ? T(1.0) : T(0.0);

        return result;
    }

    [BackwardDifferentiable]
    public T sum()
    {
        T sum = T(0.0);

        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            sum += this[i];

        return sum;
    }
}

public extension<T, int N> InlineVector<T, N> : IVector<T, N>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable {}

public extension<T, int N> InlineVector<T, N> : IVector<T, N>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [BackwardDerivative(applyBwd)]
    public OutputVector apply<int OutputSize, Storage, OutputVector>(
        Storage storage,
        no_diff Storage.Address address)
        where Storage : IStorage<T>
        where Storage.Differential == Storage
        where OutputVector : IVector<T, OutputSize>
        where OutputVector.Differential == OutputVector
    {
        var output = OutputVector(T(0.0));

        // Matrix multiplication
        [MaxIters(OutputSize)]
        for (int i = 0; i < OutputSize; i++)
        {
            let biasOffset = Storage.getOffset(address, N * OutputSize + i);
            var sum = storage.read(biasOffset);

            [ForceUnroll]
            for (int j = 0; j < N; j++)
            {
                let elementOffset = Storage.getOffset(address, j * OutputSize + i);
                sum += data[j] * storage.read(elementOffset);
            }

            output[i] = sum;
        }

        return output;
    }

    static void applyBwd<int OutputSize, Storage, OutputVector>(
        inout DifferentialPair<This> dthis,
        DifferentialPtrPair<Storage> dstorage,
        Storage.Address address,
        OutputVector.Differential doutput)
        where Storage : IStorage<T>
        where Storage.Differential == Storage
        where OutputVector : IVector<T, OutputSize>
        where OutputVector.Differential == OutputVector
    {
        // Derivative of the input is transposed weight matrix times the output differential
        var d = dthis.d;
        [MaxIters(OutputSize)]
        for (int j = 0; j < OutputSize; j++)
        {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
            {
                let elementOffset = Storage.getOffset(address, i * OutputSize + j);
                d[i] += __slang_noop_cast<OutputVector>(doutput)[j] * dstorage.p.read(elementOffset);
            }
        }

        // Derivative of the weights is the outer product of the input and the output differential
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
        {
            [ForceUnroll]
            for (int j = 0; j < OutputSize; j++)
            {
                let x = dthis.p[i];
                let dy = __slang_noop_cast<OutputVector>(doutput)[j];
                let elementOffset = Storage.getOffset(address, i * OutputSize + j);
                __slang_noop_cast<Storage>(dstorage.d).add(elementOffset, x * dy);
            }
        }

        // Derivative of the bias is the same as the output differential
        [ForceUnroll]
        for (int j = 0; j < OutputSize; j++)
        {
            let dout = __slang_noop_cast<OutputVector>(doutput)[j];
            let biasOffset = Storage.getOffset(address, N * OutputSize + j);
            __slang_noop_cast<Storage>(dstorage.d).add(biasOffset, dout);
        }

        dthis = DifferentialPair<This>(dthis.p, d);
    }
}