module mlp;

import neural;

public interface IMLP<T, int In, int Out>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    // public associatedtype InputVec : IVector<T, In>;
    // public associatedtype OutputVec : IVector<T, Out>;
    
    // public associatedtype Parameters : IDifferentiablePtrType;
    // public associatedtype Dual;

    // public property Parameters parameters { get; }
    // public property Dual dual { get; }

    // // public void optimize<Optimizer>(Optimizer optimizer, inout Optimizer.State state, uint tid)
    // //     where Optimizer : IOptimizer<T>;

    // public static InlineVector<float, Out> eval(Parameters parameters, InlineVector<float, In> input);
}

__include "addresses.slang";