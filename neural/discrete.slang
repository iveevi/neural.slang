implementing neural;

//////////////////
// Feature bank //
//////////////////

public struct FeatureBank<T, int Features, Storage>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where Storage : IStorage<T>
{
    Storage.Address address;

    public __init(Storage.Address address)
    {
        this.address = address;
    }

    [BackwardDerivative(evalBwd)]
    public OutputVector eval<OutputVector>(Storage storage, int index)
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

    public void evalBwd<OutputVector>(
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