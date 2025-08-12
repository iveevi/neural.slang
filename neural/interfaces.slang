implementing neural;

// Extensions for scalar types
public interface IConvertibleToInt
{
    int toInt();
}

// Storage interfaces
public interface IStorage<T> : IDifferentiablePtrType
    where T : IArithmeticAtomicable
{
    public associatedtype Address;

    public T read(Address address);
    public void add(Address address, T value);
    public void write(Address address, T value);

    public static Address getOffset(Address base, int elements);
}

public interface IBindlessStorage<T> : IDifferentiablePtrType
    where T : IArithmeticAtomicable
{
    public associatedtype BufferHandle;

    public T read();
    public void write(T value);
    public void add(T value);

    public This getOffset(int elements);

    public BufferHandle getBufferHandle();
}

// Vector interface
public interface IVector<T, int N> : /* IArithmetic, */ IDifferentiable
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    __init();
    __init(T value);

    __subscript(int index) -> T
    {
        get;
        set;
    }

    // Apply from a storage
    public OutputVector apply<int OutputSize, Storage, OutputVector>(
        Storage storage,
        no_diff Storage.Address address
    )
        where Storage : IStorage<T>
        where Storage.Differential == Storage
        where OutputVector : IVector<T, OutputSize>
        where OutputVector.Differential == OutputVector;

    // Apply from bindless storage
    public OutputVector applyBindless<int OutputSize, Storage, OutputVector>(Storage weights, Storage biases)
        where Storage : IBindlessStorage<T>
        where Storage.Differential == Storage
        where OutputVector : IVector<T, OutputSize>
        where OutputVector.Differential == OutputVector;

    // Necessary arithmetic operations
    This add(This other);
    This mul(This other);
    This sub(This other);
    This div(This other);
    This neg();

    // Specific methods required for ML operations
    This max(T other);
    This max(This other);
    This step(T threshold);
    T sum();
}

// Activation function interface
public interface IActivation<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDifferentiable]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector;
}

// Loss function interface
public interface ILoss<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDifferentiable]
    public T eval<int N, Vector>(Vector predicted, no_diff Vector expected)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector;
}

// Optimizer interface
public interface IOptimizer<T>
    where T : __BuiltinFloatingPointType
{
    public associatedtype State;

    public void step(inout State state, inout T parameter, T gradient);
}

// Encoder interface
public interface IEncoder<T, int In, int Out>
    where T : __BuiltinFloatingPointType
            where T : IArithmeticAtomicable
{
    public OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>;
}

// Learnable encoder interface
public interface ILearnableEncoder<T, int In, int Out, Storage>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
    where Storage : IStorage<T>
{
    public OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>;
}
