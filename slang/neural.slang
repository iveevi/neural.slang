//////////////////
// Library code //
//////////////////

// Storage types
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

public struct StructuredBufferStorage<T> : IStorage<T>
    where T : IArithmeticAtomicable
{
    public typealias Differential = This;

    public typealias Address = uint;
    public typealias BufferType = RWStructuredBuffer<T>;

    public BufferType buffer;

    public __init(BufferType buffer)
    {
        this.buffer = buffer;
    }
    
    public uint getParameterCount()
    {
        return buffer.getCount();
    }

    public T read(Address address)
    {
        return buffer[address];
    }

    public void add(Address address, T value)
    {
        InterlockedAdd(buffer[address], value);
    }

    public void write(Address address, T value)
    {
        buffer[address] = value;
    }

    public BufferType getBufferFromAddress(Address address)
    {
        let ptr = &buffer[address];
        return bit_cast<BufferType>(ptr);
    }

    public static Address getOffset(Address base, int elements)
    {
        return base + elements;
    }
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

struct BindlessBufferStorage<T> : IPointerLikeStorage<T>
    where T : IArithmeticAtomicable
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

        void add(T value)
        {
            InterlockedAdd(descriptor[offset], value);
        }
    }

    typealias Address = BindlessAddress;
    typealias BufferType = int;

    uint getParameterCount()
    {
        static_assert(false, "bad!");
        return 0;
    }

    T read(Address address)
    {
        static_assert(false, "bad!");
        return address.read();
    }

    void add(Address address, T value)
    {
        static_assert(false, "bad!");
    }

    void write(Address address, T value)
    {
        static_assert(false, "bad!");
    }

    BufferType getBufferFromAddress(Address address)
    {
        static_assert(false, "bad!");
        return 0;
    }

    static Address getOffset(Address base, int elements)
    {
        return Address(base.descriptor, base.offset + elements);
    }
}

// Vector types
interface IVector<T, int N> : IArithmetic, IDifferentiable
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
    
    [BackwardDifferentiable]
    OutputVector apply<int OutputSize, Storage, OutputVector>(
        Storage storage,
        no_diff Storage.Address address
    )
        where Storage : IStorage<T>
        where Storage.Differential == Storage
        where OutputVector : IVector<T, OutputSize>
        where OutputVector.Differential == OutputVector;
    
    // Specific methods required for ML operations
    [BackwardDifferentiable] This max(T other);
    [BackwardDifferentiable] This max(This other);

    This step(T threshold);

    [BackwardDifferentiable] T sum();
}

public struct InlineVector<T, int N> : IVector<T, N>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    public typealias Differential = This;

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
        [BackwardDifferentiable]
        get() { return data[index]; }

        set() { data[index] = newValue; }
    }

    // Arithmetic operations
    [BackwardDifferentiable]
    public This add(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] += this[i];

        return other;
    }
    
    [BackwardDifferentiable]
    public This sub(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] = this[i] - other[i];

        return other;
    }
    
    [BackwardDifferentiable]
    public This mul(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] *= this[i];

        return other;
    }
    
    [BackwardDifferentiable]
    public This div(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] = this[i] / other[i];

        return other;
    }
    
    [BackwardDifferentiable]
    public This mod(This other)
    {
        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            other[i] = this[i] % other[i];

        return other;
    }

    [BackwardDifferentiable]
    public This neg()
    {
        This result = this;

        [MaxIters(N)]
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

    public This step(T threshold)
    {
        This result = this;

        for (int i = 0; i < N; i++)
            result[i] = result[i] > threshold ? T(1.0) : T(0.0);

        return result;
    }

    [BackwardDifferentiable]
    public T sum()
    {
        T sum = T(0.0);

        [MaxIters(N)]
        for (int i = 0; i < N; i++)
            sum += data[i];

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
    OutputVector apply<int OutputSize, Storage, OutputVector>(
        Storage storage,
        no_diff Storage.Address address)
        where Storage : IStorage<T>
        where Storage.Differential == Storage
        where OutputVector : IVector<T, OutputSize>
        where OutputVector.Differential == OutputVector
    {
        // Assumed matrix layout is row major:
        // [w_11, w_12, ..., w_1N, b_1]
        // [w_21, w_22, ..., w_2N, b_2]
        // ...
        // [w_N1, w_N2, ..., w_NN, b_N]
        var output = OutputVector(T(0.0));

        // Matrix multiplication
        for (int i = 0; i < OutputSize; i++)
        {
            let rowOffset = Storage.getOffset(address, i * (N + 1));
            var sum = storage.read(rowOffset);

            for (int j = 0; j < N; j++)
            {
                let colOffset = Storage.getOffset(rowOffset, j);
                sum += this[j] * storage.read(colOffset);
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
        // Assumes same matrix layout as apply

        // Derivative of the input is the outer product of the output differential and the weights
        This d = This(0.0);
        for (int i = 0; i < N; i++)
        {
            T sum = T(0.0);
            for (int j = 0; j < OutputSize; j++)
            {
                let rowOffset = Storage.getOffset(address, j * (N + 1));
                let colOffset = Storage.getOffset(rowOffset, i);
                sum += doutput[j] * dstorage.p.read(colOffset);
            }

            d[i] = sum;
        }

        dthis = DifferentialPair<This>(dthis.p, d);

        // Derivative of the weights is the outer product of the input and the output differential
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < OutputSize; j++)
            {
                let rowOffset = Storage.getOffset(address, j * (N + 1));
                let colOffset = Storage.getOffset(rowOffset, i);
                let colOffsetDiff = __slang_noop_cast<Storage.Differential.Address>(colOffset);
                dstorage.d.add(colOffsetDiff, dthis.p[i] * doutput[j]);
            }
        }

        // Derivative of the bias is the same as the output differential
        for (int i = 0; i < OutputSize; i++)
        {
            let biasOffset = Storage.getOffset(address, i * (N + 1) + N);
            let biasOffsetDiff = __slang_noop_cast<Storage.Differential.Address>(biasOffset);
            dstorage.d.add(biasOffsetDiff, doutput[i]);
        }
    }
}

// Activation functions
public interface IActivation<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDifferentiable]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector;
}

public struct Identity<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDifferentiable]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        return input;
    }
}

public struct ReLU<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDerivative(evalBwd)]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        return input.max(T(0.0));
    }

    void evalBwd<int N, Vector>(inout DifferentialPair<Vector> dinput, Vector.Differential doutput)
        where Vector : IVector<T, N>
        where Vector.Differential  == Vector
    {
        let d = dinput.p.step(T(0.0)).mul(doutput);
        dinput = DifferentialPair<Vector>(dinput.p, d);
    }
}

public struct LeakyReLU<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    T alpha;

    __init(T alpha)
    {
        this.alpha = alpha;
    }

    [NoDiffThis, BackwardDerivative(evalBwd)]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        return input.max(alpha);
    }

    void evalBwd<int N, Vector>(inout DifferentialPair<Vector> dinput, Vector.Differential doutput)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        let positive = dinput.p.step(T(0.0));
        let negative = dinput.p.neg().step(T(0.0)).mul(Vector(alpha));
        let d = positive.mul(doutput) + negative.mul(doutput);
        dinput = DifferentialPair<Vector>(dinput.p, d);
    }
}

// Loss functions
public interface ILoss
{
    [NoDiffThis, BackwardDifferentiable]
    public T eval<T, int N, Vector>(Vector predicted, no_diff Vector expected)
        where T : __BuiltinFloatingPointType
        where T : IArithmeticAtomicable
        where Vector : IVector<T, N>
        where Vector.Differential == Vector;
}

public struct MeanSquaredError : ILoss
{
    [NoDiffThis, BackwardDerivative(evalBwd)]
    public T eval<T, int N, Vector>(Vector predicted, no_diff Vector expected)
        where T : __BuiltinFloatingPointType
        where T : IArithmeticAtomicable
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        let diff = predicted - expected;
        let square = diff * diff;
        return square.sum() / T(N);
    }

    void evalBwd<T, int N, Vector>(
        inout DifferentialPair<Vector> dpredicted,
        no_diff Vector expected,
        T.Differential dloss
    )
        where T : __BuiltinFloatingPointType
        where T : IArithmeticAtomicable
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        let d = (dpredicted.p - expected) * Vector(T(2.0) / T(N));
        dpredicted = DifferentialPair<Vector>(dpredicted.p, d);
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
public struct FeedForward<T, int In, int Out, Storage, Activation>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
    where Storage : IStorage<T>
    where Storage.Differential == Storage
    where Activation : IActivation<T>
{
    public Storage.Address parameters;
    public no_diff Activation activation;

    public __init(Storage.Address parameters, Activation activation)
    {
        this.parameters = parameters;
        this.activation = activation;
    }

    [BackwardDifferentiable]
    public OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>
        where OutputVector.Differential == OutputVector
    {
        let output = input.apply<Out, Storage, OutputVector>(storage, parameters);
        return activation.eval<Out, OutputVector>(output);
    }
}

// TODO: bindless storage overload

// Encoders
interface IEncoder<T, int In, int Out>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>;
}

struct SphericalHarmonicsEncoder<T, int Levels> : IEncoder<T, 3, 3 * Levels>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, 3>
        where OutputVector : IVector<T, 3 * Levels>
    {
        OutputVector output;

        return output;
    }
}

struct PositionalEncoder<T, int Dim, int Levels> : IEncoder<T, Dim, 2 * Levels * Dim>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, 2 * Levels * Dim>
    {
        OutputVector output;

        return output;
    }
}

interface ILearnableEncoder<T, int In, int Out, Storage>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
    where Storage : IStorage<T>
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>;
}

struct HashGridEncoder<T, int Dim, int Levels, int Features, Storage> : ILearnableEncoder<T, Dim, Features * Levels, Storage>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
    where Storage : IStorage<T>
{
    Storage.Address address;
    int resolution;
    int size;

    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Features * Levels>
    {
        OutputVector output;

        return output;
    }
}

// Overload for bindless storage
extension<T, int Dim, int Levels, int Features, Storage> HashGridEncoder<T, Dim, Levels, Features, Storage>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
    where Storage : IPointerLikeStorage<T>
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Features * Levels>
    {
        OutputVector output;

        return output;
    }
}