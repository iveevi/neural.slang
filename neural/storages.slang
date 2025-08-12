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

public struct BindlessBufferStorage<T> : IBindlessStorage<T>
    where T : IArithmeticAtomicable
{
    public typealias Differential = This;

    public typealias BufferHandle = RWStructuredBuffer<T>;

    public RWStructuredBuffer<T>.Handle descriptor;
    public uint offset;

    public T read()
    {
        return descriptor[offset];
    }

    public void write(T value)
    {
        descriptor[offset] = value;
    }

    public void add(T value)
    {
        InterlockedAdd(descriptor[offset], value);
    }

    public This getOffset(int elements)
    {
        return This(descriptor, offset + elements);
    }

    public BufferHandle getBufferHandle()
    {
        return descriptor;
    }
}