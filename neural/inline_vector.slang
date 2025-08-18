implementing neural;

////////////////////
// Inline vectors //
////////////////////

public struct InlineVector<T, int N> : IVector<T, N>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    public static const int Dimension = N;

    public typealias Differential = This;

    internal T[N] data;
    
    public __init() {}
    
    public __init(int value) {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            data[i] = T(value);
    }
    
    public __init(T value) {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            data[i] = value;
    }

    public __init(T[N] data) { this.data = data; }

    public __init(This other) { this.data = other.data; }

    public __subscript(int index) -> T
    {
        [BackwardDifferentiable]
        get() { return getIndex(index); }

        // TODO: make this differentiable
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
    public static Differential dzero()
    {
        return This(T(0.0));
    }

    public static Differential dadd(Differential a, Differential b)
    {
        return a.add(b);
    }

    public static Differential dmul(Differential a, Differential b)
    {
        return a.mul(b);
    }

    // Necessary arithmetic operations
    [BackwardDerivative(addBwd)]
    public This add(This other)
    {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            other.data[i] = this.data[i] + other.data[i];

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
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            other.data[i] = this.data[i] - other.data[i];

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
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            other.data[i] = this.data[i] * other.data[i];

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
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            other.data[i] = this.data[i] / other.data[i];

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

    public This neg()
    {
        This result = this;

        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = -this.data[i];

        return result;
    }
    
    // Additional ML operations
    public This max(T other)
    {
        This result = this;

        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = max(result.data[i], other);

        return result;
    }
    
    public This max(This other)
    {
        This result = this;

        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = max(result.data[i], other.data[i]);

        return result;
    }

    public This step(T threshold)
    {
        This result = this;

        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = (result.data[i] > threshold) ? T(1.0) : T(0.0);

        return result;
    }

    // TODO: start relying more on autodiff and remove the explicit bwds
    [BackwardDerivative(sinBwd)]
    public This sin()
    {
        This result = this;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = sin(result.data[i]);

        return result;
    }

    static void sinBwd(
        inout DifferentialPair<This> dthis,
        Differential doutput
    )
    {
        dthis = DifferentialPair<This>(dthis.p, dmul(dthis.p.cos(), doutput).add(dthis.d));
    }

    [BackwardDerivative(cosBwd)]
    public This cos()
    {
        This result = this;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = cos(result.data[i]);

        return result;
    }

    static void cosBwd(
        inout DifferentialPair<This> dthis,
        Differential doutput
    )
    {
        dthis = DifferentialPair<This>(dthis.p, dmul(dthis.p.sin().neg(), doutput).add(dthis.d));
    }

    [BackwardDerivative(absBwd)]
    public This abs()
    {
        This result = this;

        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = abs(result.data[i]);

        return result;
    }

    static void absBwd(

        inout DifferentialPair<This> dthis,
        Differential doutput
    )
    {
        dthis = DifferentialPair<This>(dthis.p, dmul(dthis.p.sign(), doutput).add(dthis.d));
    }

    [BackwardDerivative(expBwd)]
    public This exp()
    {
        This result = this;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = exp(result.data[i]);

        return result;
    }

    static void expBwd(
        inout DifferentialPair<This> dthis,
        Differential doutput
    )
    {
        dthis = DifferentialPair<This>(dthis.p, dmul(dthis.p.exp(), doutput).add(dthis.d));
    }

    public This sign()
    {
        This result = this;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = (result.data[i] > T(0.0)) ? T(1.0) : T(-1.0);

        return result;
    }

    [BackwardDerivative(sumBwd)]
    public T sum()
    {
        T sum = T(0.0);

        [ForceUnroll]
        for (int i = 0; i < N; i++)
            sum += this.data[i];

        return sum;
    }

    static void sumBwd(
        inout DifferentialPair<This> dthis,
        T.Differential doutput)
    {
        dthis = DifferentialPair<This>(
            dthis.p,
            dadd(dthis.d, This(__slang_noop_cast<T>(doutput)))
        );
    }

    // Interpolation
    [BackwardDifferentiable]
    public static This interpolate(This a, This b, T t)
    {
        This result = This();
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result[i] = a[i] * (T(1.0) - t) + b[i] * t;

        return result;
    }
}

public extension<T, int N> InlineVector<T, N> : IVector<T, N>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    // Conversion to and from vectors
    [BackwardDerivative(toVectorBwd)]
    public vector<T, N> toVector()
    {
        vector<T, N> result;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result[i] = this.data[i];

        return result;
    }

    static void toVectorBwd(
        inout DifferentialPair<This> dthis,
        vector<T, N> doutput
    )
    {
        dthis = DifferentialPair<This>(dthis.p, fromVector(doutput));
    }

    [BackwardDerivative(fromVectorBwd)]
    public static This fromVector(vector<T, N> v)
    {
        This result = This();
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = v[i];

        return result;
    }

    static void fromVectorBwd(
        inout DifferentialPair<vector<T, N>> dv,
        This doutput
    )
    {
        dv = DifferentialPair<vector<T, N>>(dv.p, doutput.toVector());
    }

    // Slicing
    [BackwardDerivative(sliceBwd)]
    public InlineVector<T, M> slice<int K, int M>()
    {
        static_assert(K >= 0 && M + K <= N, "Invalid slice!");

        var result = InlineVector<T, M>();
        [ForceUnroll]
        for (int i = 0; i < M; i++)
            result.data[i] = data[i + K];

        return result;
    }

    public static void sliceBwd<int K, int M>(
        inout DifferentialPair<This> dthis,
        InlineVector<T, M>.Differential doutput
    )
    {
        var d = dthis.d;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            d[i + K] += doutput[i];

        dthis = DifferentialPair<This>(dthis.p, d);
    }
}

// Ordinary storage
public extension<T, int N> InlineVector<T, N> : IVector<T, N>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
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

        [ForceUnroll]
        for (int i = 0; i < OutputSize; i++)
        {
            let biasOffset = Storage.getOffset(address, N * OutputSize + i);
            output[i] = storage.read(biasOffset);
        }

        // TODO: transpose if N >> OutputSize or OutputSize < 4
        [MaxIters(N)]
        for (int j = 0; j < N; j++)
        {
            let x = data[j];
            [ForceUnroll]
            for (int i = 0; i < OutputSize; i++)
            {
                let elementOffset = Storage.getOffset(address, j * OutputSize + i);
                output[i] += x * storage.read(elementOffset);
            }
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
        let doutputVector = __slang_noop_cast<OutputVector>(doutput);

        // Derivative of the input is transposed weight matrix times the output differential
        var d = dthis.d;
        [MaxIters(OutputSize)]
        for (int j = 0; j < OutputSize; j++)
        {
            let dy = doutputVector[j];

            [ForceUnroll]
            for (int i = 0; i < N; i++)
            {
                let elementOffset = Storage.getOffset(address, i * OutputSize + j);
                d[i] += dy * dstorage.p.read(elementOffset);
            }
        }

        // Derivative of the weights is the outer product of the input and the output differential
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
        {
            let x = dthis.p[i];
            [ForceUnroll]
            for (int j = 0; j < OutputSize; j++)
            {
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

// Bindless storage
public extension<T, int N> InlineVector<T, N> : IVector<T, N>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    [BackwardDerivative(applyBindlessBwd)]
    public OutputVector applyBindless<int OutputSize, Storage, OutputVector>(Storage weights, Storage biases)
        where Storage : IBindlessStorage<T>
        where Storage.Differential == Storage
        where OutputVector : IVector<T, OutputSize>
        where OutputVector.Differential == OutputVector
    {
        var output = OutputVector();

        [ForceUnroll]
        for (int i = 0; i < OutputSize; i++)
            output[i] = biases.getOffset(i).read();

        [MaxIters(N)]
        for (int j = 0; j < N; j++)
        {
            let x = data[j];
            [ForceUnroll]
            for (int i = 0; i < OutputSize; i++)
                output[i] += x * weights.getOffset(j * OutputSize + i).read();
        }

        return output;
    }

    static void applyBindlessBwd<int OutputSize, Storage, OutputVector>(
        inout DifferentialPair<This> dthis,
        DifferentialPtrPair<Storage> dweights,
        DifferentialPtrPair<Storage> dbiases,
        OutputVector.Differential doutput)
        where Storage : IBindlessStorage<T>
        where Storage.Differential == Storage
        where OutputVector : IVector<T, OutputSize>
        where OutputVector.Differential == OutputVector
    {
        let doutputVector = __slang_noop_cast<OutputVector>(doutput);

        // Derivative of the input is transposed weight matrix times the output differential
        var d = dthis.d;
        [MaxIters(OutputSize)]
        for (int j = 0; j < OutputSize; j++)
        {
            let dy = doutputVector[j];
        
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                d[i] += dy * dweights.p.getOffset(i * OutputSize + j).read();
        }

        // Derivative of the weights is the outer product of the input and the output differential
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
        {
            let x = dthis.p[i];
            [ForceUnroll]
            for (int j = 0; j < OutputSize; j++)
            {
                let dy = __slang_noop_cast<OutputVector>(doutput)[j];
                let offset = i * OutputSize + j;
                __slang_noop_cast<Storage>(dweights.d).getOffset(offset).add(x * dy);
            }
        }

        // Derivative of the bias is the same as the output differential
        [ForceUnroll]
        for (int j = 0; j < OutputSize; j++)
        {
            let dout = __slang_noop_cast<OutputVector>(doutput)[j];
            let offset = N * OutputSize + j;
            __slang_noop_cast<Storage>(dbiases.d).getOffset(offset).add(dout);
        }

        dthis = DifferentialPair<This>(dthis.p, d);
    }
}