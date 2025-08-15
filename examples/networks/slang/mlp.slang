module mlp;

import neural;

public interface IMLP<T, int In, int Out>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    public associatedtype InputVec : IVector<T, In>;
    public associatedtype OutputVec : IVector<T, Out>;
    
    public associatedtype Parameters : IDifferentiablePtrType;
    public associatedtype Configuration;

    public property Parameters parameters { get; }
    public property Configuration configuration { get; }
    public property DifferentialPtrPair<Parameters> parametersDual { get; }

    [BackwardDifferentiable]
    public static OutputVec eval(Parameters parameters, Configuration configuration, InputVec input);
}

__include "addresses.slang";