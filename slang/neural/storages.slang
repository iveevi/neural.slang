implementing neural;

////////////////////////////////
// RWStructuredBuffer storage //
////////////////////////////////

public struct StructuredBufferStorage<T> : IStorage<T>
{
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

    // public void add(Address address, T value)
    // {
    //     buffer[address] += value;
    // }

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

public extension<T> StructuredBufferStorage<T> : IParameterStorage<T>
    where T : IArithmeticAtomicable
{
    public typealias Differential = This;

    public void add(Address address, T value)
    {
        InterlockedAdd(buffer[address], value);
    }
}

/////////////////////////////
// Bindless buffer storage //
/////////////////////////////

// struct BindlessBufferStorage<T> : IPointerLikeStorage<T>
//     where T : IArithmeticAtomicable
// {
//     typealias Differential = This;

//     struct BindlessAddress : IPointerLikeAddress<T>
//     {
//         RWStructuredBuffer<T>.Handle descriptor;
//         uint offset;

//         T read()
//         {
//             return descriptor[offset];
//         }

//         void write(T value)
//         {
//             descriptor[offset] = value;
//         }

//         void add(T value)
//         {
//             InterlockedAdd(descriptor[offset], value);
//         }
//     }

//     typealias Address = BindlessAddress;
//     typealias BufferType = int;

//     T read(Address address)
//     {
//         static_assert(false, "Not implemented");
//         return address.read();
//     }

//     void add(Address address, T value)
//     {
//         static_assert(false, "Not implemented");
//     }

//     void write(Address address, T value)
//     {
//         static_assert(false, "Not implemented");
//     }

//     BufferType getBufferFromAddress(Address address)
//     {
//         static_assert(false, "Not implemented");
//         return 0;
//     }

//     static Address getOffset(Address base, int elements)
//     {
//         return Address(base.descriptor, base.offset + elements);
//     }
// }