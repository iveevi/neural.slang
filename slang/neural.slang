//////////////////
// Library code //
//////////////////

// Storage types
interface IStorage<T> : IDifferentiablePtrType
{
    // Iterator for storage pameters
    associatedtype Address;

    // Handle for matrix multiplication, likely
    // needs to conform to an appopriate interface 
    associatedtype BufferType;

    T read(Address address);
    void write(Address address, T value);

    Address getBaseAddress();
    BufferType getBufferFromAddress(Address address);

    static Address getOffset(Address base, int elements);
}

struct StructuredBufferStorage<T> : IStorage<T>
{
    typealias Differential = This;

    typealias Address = uint;
    typealias BufferType = RWStructuredBuffer<T>;

    BufferType buffer;
    
    uint getParameterCount()
    {
        return buffer.getCount();
    }

    T read(Address address)
    {
        return buffer[address];
    }

    void write(Address address, T value)
    {
        buffer[address] = value;
    }

    Address getBaseAddress()
    {
        return 0;
    }

    BufferType getBufferFromAddress(Address address)
    {
        let ptr = &buffer[address];
        return bit_cast<BufferType>(ptr);
    }

    static Address getOffset(Address base, int elements)
    {
        return base + elements;
    }
}

interface IPointerLikeAddress<T>
{
    T read();
    void write(T value);
}

interface IPointerLikeStorage<T> : IStorage<T>
{
    associatedtype Address : IPointerLikeAddress<T>;
}

struct BindlessBufferStorage<T> : IPointerLikeStorage<T>
{
    typealias Differential = This;

    struct BindlessAddress : IPointerLikeAddress<T>
    {
        RWStructuredBuffer<T>.Handle descriptor;
        uint offset;

        T read()
        {
            return descriptor[offset];
        }

        void write(T value)
        {
            descriptor[offset] = value;
        }
    }

    typealias Address = BindlessAddress;
    typealias BufferType = int;

    uint getParameterCount()
    {
        static_assert(false, "bad!");
    }

    T read(Address address)
    {
        static_assert(false, "bad!");
    }

    void write(Address address, T value)
    {
        static_assert(false, "bad!");
    }

    Address getBaseAddress()
    {
        static_assert(false, "bad!");
    }

    BufferType getBufferFromAddress(Address address)
    {
        static_assert(false, "bad!");
    }

    static Address getOffset(Address base, int elements)
    {
        return Address(base.descriptor, base.offset + elements);
    }
}

// Vector types
interface IVector<T, int N> : IArithmetic, IDifferentiable
    where T : __BuiltinFloatingPointType
{
    __init();
    __init(T value);

    __subscript(int index) -> T
    {
        get;
        set;
    }
    
    [BackwardDifferentiable]
    OutputVector apply<int OutputSize, Storage, OutputVector>(
        Storage storage,
        no_diff Storage.Address address
    )
        where Storage : IStorage<T>
        where OutputVector : IVector<T, OutputSize>;
    
    // Specific methods required for ML operations
    [BackwardDifferentiable] This max(T other);
    [BackwardDifferentiable] This max(This other);
}

public struct InlineVector<T, int N>
    where T : __BuiltinFloatingPointType
{
    internal T[N] data;
    
    public __init() {}
    public __init(int value) { data[0] = T(value); }
    public __init(T value) { data[0] = value; }
    public __init(T[N] data) { this.data = data; }

    public __init(vector<T, N> data)
    {
        for (int i = 0; i < N; i++)
            this.data[i] = data[i];
    }

    public __init(This other) { this.data = other.data; }

    public __subscript(int index) -> T
    {
        get() { return data[index]; }
        set() { data[index] = newValue; }
    }

    // Arithmetic operations
    [BackwardDifferentiable]
    public This add(This other)
    {
        for (int i = 0; i < N; i++)
            other[i] += this[i];
    }
    
    [BackwardDifferentiable]
    public This sub(This other)
    {
        for (int i = 0; i < N; i++)
            other[i] = this[i] - other[i];
    }
    
    [BackwardDifferentiable]
    public This mul(This other)
    {
        for (int i = 0; i < N; i++)
            other[i] *= this[i];
    }
    
    [BackwardDifferentiable]
    public This div(This other)
    {
        for (int i = 0; i < N; i++)
            other[i] = this[i] / other[i];
    }
    
    [BackwardDifferentiable]
    public This mod(This other)
    {
        for (int i = 0; i < N; i++)
            other[i] = this[i] % other[i];
    }
    
    [BackwardDifferentiable]
    public This neg()
    {
        This result = this;
        for (int i = 0; i < N; i++)
            result[i] = -result[i];

        return result;
    }

    // Comparison operations
    public bool equals(This other)
    {
        for (int i = 0; i < N; i++)
            if (this[i] != other[i]) return false;
        return true;
    }
    
    public bool lessThan(This other)
    {
        for (int i = 0; i < N; i++)
            if (this[i] >= other[i]) return false;
        return true;
    }
    
    public bool lessThanOrEquals(This other)
    {
        for (int i = 0; i < N; i++)
            if (this[i] > other[i]) return false;
        return true;
    }
    
    // Additional ML operations
    [BackwardDifferentiable]
    public This max(T other)
    {
        This result = this;
        for (int i = 0; i < N; i++)
            result[i] = max(result[i], other);

        return result;
    }
    
    [BackwardDifferentiable]
    public This max(This other)
    {
        This result = this;
        for (int i = 0; i < N; i++)
            result[i] = max(result[i], other[i]);

        return result;
    }
}

public extension<T, int N> InlineVector<T, N> : IVector<T, N>
    where T : __BuiltinFloatingPointType {}

public extension<T, int N> InlineVector<T, N> : IVector<T, N>
    where T : __BuiltinFloatingPointType
{
    [BackwardDerivative(applyBwd)]
    OutputVector apply<int OutputSize, Storage, OutputVector>(
        Storage storage,
        no_diff Storage.Address address)
        where Storage : IStorage<T>
        where OutputVector : IVector<T, OutputSize>
    {
        var output = OutputVector();

        // Matrix multiplication
        for (int i = 0; i < OutputSize; i++)
        {
            var sum = T(0.0);
            for (int j = 0; j < N; j++)
            {
                let offset = Storage.getOffset(address, j);
                sum += this[j] * storage.read(offset);
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
        where OutputVector : IVector<T, OutputSize>
    {
        static_assert(false, "bad!");
        // // Weights differential is the outer product of the input and the output differential
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < OutputSize; j++)
        //     {
        //         let offset = Storage.getOffset(address, i);
        //         let dvalue = doutput[j];
        //         dstorage.write(offset, dvalue * this[i]);
        //     }
        // }
    }
}

// Activation functions
public interface IActivation<T>
    where T : __BuiltinFloatingPointType
{
    [NoDiffThis, BackwardDifferentiable]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>;
}

public struct Identity<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
{
    [NoDiffThis, BackwardDifferentiable]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
    {
        return input;
    }
}

public struct ReLU<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
{
    [NoDiffThis, BackwardDifferentiable]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
    {
        return input.max(T(0.0));
    }
}

public struct LeakyReLU<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
{
    T alpha;

    __init(T alpha)
    {
        this.alpha = alpha;
    }

    [NoDiffThis, BackwardDifferentiable]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
    {
        return input.max(alpha);
    }
}

// Loss functions
interface ILoss
{
    [NoDiffThis, BackwardDifferentiable]
    T eval<VectorType, T, int N>(VectorType predicted, VectorType expected)
        where T : __BuiltinFloatingPointType
        where VectorType : IVector<T, N>;
}

struct MeanSquaredError : ILoss
{
    [NoDiffThis, BackwardDifferentiable]
    T eval<VectorType, T, int N>(VectorType predicted, VectorType expected)
        where T : __BuiltinFloatingPointType
        where VectorType : IVector<T, N>
    {
        let diff = predicted - expected;
        let square = diff * diff;

        var sum = T(0.0);

        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            sum += square[i];

        return sum / T(N);
    }
}

// Optimizers
interface IOptimizer<T>
    where T : __BuiltinFloatingPointType
{
    associatedtype State;

    void step(inout State state, inout T parameter, T gradient);
}

struct Adam<T> : IOptimizer<T>
    where T : __BuiltinFloatingPointType
{
    struct AdamState {
        T m;
        T v;
        int t;
    };

    typedef AdamState State;

    T alpha;
    T beta1;
    T beta2;
    T epsilon;

    void step(inout State state, inout T parameter, T gradient)
    {
        state.t += 1;
        state.m = beta1 * state.m + (T(1.0) - beta1) * gradient;
        state.v = beta2 * state.v + (T(1.0) - beta2) * gradient * gradient;

        T m_hat = state.m / (T(1.0) - pow(beta1, T(state.t)));
        T v_hat = state.v / (T(1.0) - pow(beta2, T(state.t)));

        parameter -= alpha * m_hat / (sqrt(v_hat) + epsilon);
    }
}

// Feed forward layer
struct FeedForward<T, int In, int Out, Storage, Activation>
    where T : __BuiltinFloatingPointType
    where Storage : IStorage<T>
    where Activation : IActivation<T>
{
    Storage.Address parameters;
    no_diff Activation activation;

    __init(Storage.Address parameters, Activation activation)
    {
        this.parameters = parameters;
        this.activation = activation;
    }

    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>
    {
        let out = input.apply<Out, Storage, OutputVector>(storage, parameters);
        return activation.eval(out);
    }
}

// Encoders
interface IEncoder<T, int In, int Out>
    where T : __BuiltinFloatingPointType
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>;
}

struct SphericalHarmonicsEncoder<T, int Levels> : IEncoder<T, 3, 3 * Levels>
    where T : __BuiltinFloatingPointType
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, 3>
        where OutputVector : IVector<T, 3 * Levels> { /* ... */ }
}

struct PositionalEncoder<T, int Dim, int Levels> : IEncoder<T, Dim, 2 * Levels * Dim>
    where T : __BuiltinFloatingPointType
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, 2 * Levels * Dim> { /* ... */ }
}

interface ILearnableEncoder<T, int In, int Out, Storage>
    where T : __BuiltinFloatingPointType
    where Storage : IStorage<T>
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>;
}

struct HashGridEncoder<T, int Dim, int Levels, int Features, Storage> : ILearnableEncoder<T, Dim, Features * Levels, Storage>
    where T : __BuiltinFloatingPointType
    where Storage : IStorage<T>
{
    Storage.Address address;
    int resolution;
    int size;

    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Features * Levels> { /* ... */ }
}

// Overload for bindless storage
extension<T, int Dim, int Levels, int Features, Storage> HashGridEncoder<T, Dim, Levels, Features, Storage>
    where T : __BuiltinFloatingPointType
    where Storage : IPointerLikeStorage<T>
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Features * Levels> { /* ... */ }
}