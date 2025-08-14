 interface IStorage<T> : IDifferentiablePtrType
    where T : IArithmeticAtomicable
{
     associatedtype Address;

     T read(Address address);
     void add(Address address, T value);

     static Address getOffset(Address base, int elements);
}

 struct StructuredBufferStorage<T> : IStorage<T>
    where T : IArithmeticAtomicable
{
     typealias Differential = This;

     typealias Address = uint;
     typealias BufferType = RWStructuredBuffer<T>;

     BufferType buffer;

     __init(BufferType buffer)
    {
        this.buffer = buffer;
    }

     T read(Address address)
    {
        return buffer[address];
    }

     void add(Address address, T value)
    {
        InterlockedAdd(buffer[address], value);
    }

     static Address getOffset(Address base, int elements)
    {
        return base + elements;
    }
}

 interface IVector<T, int N> : IDifferentiable
    where T : __BuiltinFloatingPointType
{
    __init(T value);

    __subscript(int index) -> T
    {
        get;
        set;
    }
}

 struct InlineVector<T, int N> : IVector<T, N>
    where T : __BuiltinFloatingPointType
{
     static const int Dimension = N;

     typealias Differential = This;

    internal T[N] data;
    
     __init(T value) {
        for (int i = 0; i < N; i++)
            data[i] = value;
    }

     __subscript(int index) -> T
    {
        get() { return data[index]; }
        set() { data[index] = newValue; }
    }
}

 struct FeatureBank<T, int Features, Storage>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where Storage : IStorage<T>
{
    Storage.Address address;

     __init(Storage.Address address)
    {
        this.address = address;
    }

    [BackwardDerivative(evalBwd)]
     OutputVector eval<OutputVector>(Storage storage, int index)
        where OutputVector : IVector<T, Features>
        where OutputVector.Differential == OutputVector
    {
        var output = OutputVector(T(0.0));

        [ForceUnroll]
        for (int i = 0; i < Features; i++)
        {
            let offset = Storage.getOffset(address, index * Features + i);
            output[i] = storage.read(offset);
        }

        return output;
    }

     void evalBwd<OutputVector>(
        DifferentialPtrPair<Storage> dstorage,
        int index,
        OutputVector.Differential doutput
    )
        where OutputVector : IVector<T, Features>
        where OutputVector.Differential == OutputVector
    {
        [ForceUnroll]
        for (int i = 0; i < Features; i++)
        {
            let offset = Storage.getOffset(address, index * Features + i);
            __slang_noop_cast<Storage>(dstorage.d).add(offset, doutput[i]);
        }
    }
}

static const int Features = 8;

typealias OutputVec = InlineVector<float, Features>;
typealias Storage = StructuredBufferStorage<float>;

struct FeatureBankDerivativeGlobals
{
    RWStructuredBuffer<float> parameters;
    RWStructuredBuffer<float> dparameters;
    RWStructuredBuffer<uint> indices;
    uint count;
}

ParameterBlock<FeatureBankDerivativeGlobals> globals;

[BackwardDifferentiable]
OutputVec feature_bank_eval(Storage storage, int index)
{
    let bank = FeatureBank<float, Features, Storage>(0);
    return bank.eval<OutputVec>(storage, index);
}

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain(uint3 thread_id : SV_DispatchThreadID)
{
    int tid = thread_id.x;
    if (tid >= globals.count) return;

    let index = int(globals.indices[tid]);
    var storage = Storage(globals.parameters);
    var dstorage = Storage(globals.dparameters);
    var storageDiff = DifferentialPtrPair<Storage>(storage, dstorage);
    
    bwd_diff(feature_bank_eval)(storageDiff, index, OutputVec(1.0));
}
