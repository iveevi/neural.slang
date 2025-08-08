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
        for (int i = 0; i < OutputSize; i++)
        {
            let biasOffset = Storage.getOffset(address, N * OutputSize + i);
            var sum = storage.read(biasOffset);

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
        for (int j = 0; j < OutputSize; j++)
        {
            for (int i = 0; i < N; i++)
            {
                let elementOffset = Storage.getOffset(address, i * OutputSize + j);
                d[i] += __slang_noop_cast<OutputVector>(doutput)[j] * dstorage.p.read(elementOffset);
            }
        }

        // Derivative of the weights is the outer product of the input and the output differential
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < OutputSize; j++)
            {
                let x = dthis.p[i];
                let dy = __slang_noop_cast<OutputVector>(doutput)[j];
                let elementOffset = Storage.getOffset(address, i * OutputSize + j);
                __slang_noop_cast<Storage>(dstorage.d).add(elementOffset, x * dy);
            }
        }

        // Derivative of the bias is the same as the output differential
        for (int j = 0; j < OutputSize; j++)
        {
            let dout = __slang_noop_cast<OutputVector>(doutput)[j];
            let biasOffset = Storage.getOffset(address, N * OutputSize + j);
            __slang_noop_cast<Storage>(dstorage.d).add(biasOffset, dout);
        }

        dthis = DifferentialPair<This>(dthis.p, d);
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
public interface ILoss<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDifferentiable]
    public T eval<int N, Vector>(Vector predicted, no_diff Vector expected)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector;
}

public struct MeanSquaredError<T> : ILoss<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDifferentiable]
    public T eval<int N, Vector>(Vector predicted, no_diff Vector expected)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        let diff = predicted.sub(expected);
        let square = diff.mul(diff);
        return square.sum() / T(N);
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

    [NoDiffThis, BackwardDifferentiable]
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

struct SphericalHarmonicsEncoder<T, int Levels> : IEncoder<T, 3, 2 * Levels + 1>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, 3>
        where OutputVector : IVector<T, 2 * Levels + 1>
    {
        OutputVector output;

        return output;
    }
}

public struct FrequencyEncoder<T, int Dim, int Levels> : IEncoder<T, Dim, 2 * Levels * Dim>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [BackwardDerivative(evalBwd)]
    public OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, 2 * Levels * Dim>
    {
        OutputVector output;

        [ForceUnroll]
        for (int i = 0; i < Dim; i++)
        {
            T k = T(1.0);

            for (int j = 0; j < Levels; j++)
            {
                let frequency = k * T.getPi();
                let sin = sin(frequency * input[i]);
                let cos = cos(frequency * input[i]);
                output[2 * j * Dim + i] = sin;
                output[2 * j * Dim + i + Dim] = cos;
                k *= T(2.0);
            }
        }

        return output;
    }

    void evalBwd<InputVector, OutputVector>(inout DifferentialPair<InputVector> dinput, OutputVector.Differential doutput)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, 2 * Levels * Dim>
    {
        let p = __slang_noop_cast<InputVector>(dinput.p);
        var d = __slang_noop_cast<InputVector>(dinput.d);
        let dout = __slang_noop_cast<OutputVector>(doutput);

        [ForceUnroll]
        for (int i = 0; i < Dim; i++)
        {
            T k = T(1.0);
            for (int j = 0; j < Levels; j++)
            {
                let frequency = k * T.getPi();
                let sinVal = sin(frequency * p[i]);
                let cosVal = cos(frequency * p[i]);
                let dsinOut = dout[2 * j * Dim + i];
                let dcosOut = dout[2 * j * Dim + i + Dim];
                d[i] += dsinOut * cosVal * frequency;
                d[i] += dcosOut * -sinVal * frequency;
                k *= T(2.0);
            }
        }

        dinput = DifferentialPair<InputVector>(p, __slang_noop_cast<InputVector.Differential>(d));
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