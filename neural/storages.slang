implementing neural;

////////////////////////////////
// RWStructuredBuffer storage //
////////////////////////////////

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

/////////////////////////////
// Bindless buffer storage //
/////////////////////////////

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