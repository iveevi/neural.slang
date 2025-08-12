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
    // Iterator for storage parameters
    public associatedtype Address;

    // Handle for matrix multiplication, likely
    // needs to conform to an appopriate interface
    public associatedtype BufferType;

    public T read(Address address);
    public void add(Address address, T value);
    public void write(Address address, T value);

    public BufferType getBufferFromAddress(Address address);

    public static Address getOffset(Address base, int elements);
}

interface IPointerLikeAddress<T>
{
    T read();
    void write(T value);
    void add(T value);
}

interface IPointerLikeStorage<T> : IStorage<T>
    where T : IArithmeticAtomicable
{
    associatedtype Address : IPointerLikeAddress<T>;
}

// Vector interface
public interface IVector<T, int N> : IArithmetic, IDifferentiable
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

    public OutputVector apply<int OutputSize, Storage, OutputVector>(
        Storage storage,
        no_diff Storage.Address address
    )
        where Storage : IStorage<T>
        where Storage.Differential == Storage
        where OutputVector : IVector<T, OutputSize>
        where OutputVector.Differential == OutputVector;

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
    [BackwardDifferentiable]
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
    [BackwardDifferentiable]
    public OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>;
}
