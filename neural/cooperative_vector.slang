implementing neural;

/////////////////////////
// Cooperative vectors //
/////////////////////////

public struct CoopVector<T, int N> : IVector<T, N>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    public static const int Dimension = N;

    public typealias Differential = This;

    internal CoopVec<T, N> data;
    
    public __init() {}
    
    public __init(int value) {
        data.fill(T(value));
    }
    
    public __init(T value) {
        data.fill(value);
    }

    public __init(CoopVec<T, N> data) {
        this.data.copyFrom(data);
    }

    public __init(This other) {
        this.data.copyFrom(other.data);
    }

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

    // Arithmetic operations
    [BackwardDerivative(addBwd)]
    public This add(This other)
    {
        return This(this.data + other.data);
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
        return This(this.data - other.data);
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
        return This(this.data * other.data);
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
        return This(this.data / other.data);
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
        return This(-this.data);
    }
    
    // Additional ML operations
    public This max(T other)
    {
        return This(max(this.data, CoopVec<T, N>(other)));
    }
    
    public This max(This other)
    {
        return This(max(this.data, other.data));
    }

    public This step(T threshold)
    {
        var result = CoopVec<T, N>();
        for (int i = 0; i < N; i++)
            result[i] = (data[i] > threshold) ? T(1.0) : T(0.0);
        return This(result);
    }

    [BackwardDerivative(sumBwd)]
    public T sum()
    {
        var sum = T(0.0);
        for (int i = 0; i < N; i++)
            sum += this.data[i];
        return sum;
    }

    static void sumBwd(
        inout DifferentialPair<This> dthis,
        T.Differential doutput
    )
    {
        dthis = DifferentialPair<This>(
            dthis.p,
            dadd(dthis.d, This(__slang_noop_cast<T>(doutput)))
        );
    }
}

// Ordinary storage
public extension<T, int N> CoopVector<T, N> : IVector<T, N>
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
        static_assert(false, "Not implemented");
        return OutputVector(T(0.0));
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
        static_assert(false, "Not implemented");
    }
}

// Bindless storage
static CoopVecComponentType getCoopVecComponentType<T>()
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    if (T is half)
        return CoopVecComponentType.Float16;
    else if (T is float)
        return CoopVecComponentType.Float32;
    else
        static_assert(false, "Unsupported type");

    return CoopVecComponentType.Float32;
}

public extension<T, int N> CoopVector<T, N> : IVector<T, N>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    static const CoopVecComponentType ComponentType = getCoopVecComponentType<T>();

    [BackwardDerivative(applyBindlessBwd)]
    public OutputVector applyBindless<int OutputSize, Storage, OutputVector>(Storage weights, Storage biases)
        where Storage : IBindlessStorage<T>
        where Storage.Differential == Storage
        where OutputVector : IVector<T, OutputSize>
        where OutputVector.Differential == OutputVector
    {
        let weightsStorage = __slang_noop_cast<RWStructuredBuffer<T>>(weights.getBufferHandle());
        let biasesStorage = __slang_noop_cast<RWStructuredBuffer<T>>(biases.getBufferHandle());
        let output = coopVecMatMulAdd<T, OutputSize>(
            this.data, ComponentType,
            weightsStorage, 0, ComponentType,
            biasesStorage, 0, ComponentType,
            CoopVecMatrixLayout.RowMajor,
            false,
            N * sizeof(T)
        );

        return __slang_noop_cast<OutputVector>(CoopVector<T, OutputSize>(output));
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
        let weightsGradStorage = __slang_noop_cast<RWStructuredBuffer<T>>(dweights.d.getBufferHandle());
        let biasesGradStorage = __slang_noop_cast<RWStructuredBuffer<T>>(dbiases.d.getBufferHandle());
        let weightsStorage = __slang_noop_cast<RWStructuredBuffer<T>>(dweights.p.getBufferHandle());
        let outputGrad = __slang_noop_cast<CoopVector<T, OutputSize>>(doutput).data;

        // Derivative of the weight matrix
        coopVecOuterProductAccumulate(
            outputGrad,
            dthis.d.data,
            weightsGradStorage, 0,
            0,
            CoopVecMatrixLayout.TrainingOptimal,
            ComponentType
        );
        
        // Derivative of the bias vector
        coopVecReduceSumAccumulate(outputGrad, biasesGradStorage, 0);

        // Derivative of the input vector
        let d = coopVecMatMul<T, N>(
            outputGrad, ComponentType,
            weightsStorage, 0, ComponentType,
            CoopVecMatrixLayout.ColumnMajor,
            false,
            N * sizeof(T)
        );

        dthis = DifferentialPair<This>(dthis.p, This(d));
    }
}