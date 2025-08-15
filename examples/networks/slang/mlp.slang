module networks;

import neural;

public interface IMLP<T, int In, int Out>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    public associatedtype Parameters;
    public associatedtype Configuration;

    public associatedtype InputVec : IVector<T, In>;
    public associatedtype OutputVec : IVector<T, Out>;

    public property Parameters parameters { get; }
    public property Configuration configuration { get; }

    [BackwardDifferentiable]
    public static OutputVec eval(Parameters parameters, Configuration configuration, InputVec input);
}