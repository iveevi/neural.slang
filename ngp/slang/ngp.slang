module ngp;

import neural;

public interface IOptimizable<T>
    where T : __BuiltinFloatingPointType
{
    public associatedtype Parameters;
    public associatedtype Dual;

    public property Parameters parameters { get; }
    public property Dual dual { get; }

    public void update<Optimizer>(Optimizer optimizer, inout Optimizer.State state, uint tid)
        where Optimizer : IOptimizer<T>;
}

__include "address_based_mlp.slang";
__include "dense_grid_impl.slang";
__include "dense_grid.slang";
__include "multi_level_dense_grid.slang";