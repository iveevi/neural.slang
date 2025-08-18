module dense_grid;

import neural;

// Single level feature grid
// TODO: spit into dense_grid_impl.slang and dense_grid.slang
public struct DenseGridImpl<int Dimension, int Features>
{
    public typealias FeatureVector = InlineVector<float, Features>;

    int offset;
    int resolution;

    public __init(int offset, int resolution)
    {
        this.offset = offset;
        this.resolution = resolution;
    }

    public property int stride
    {
        get {
            var result = 1;
            [ForceUnroll]
            for (int i = 0; i < Dimension; i++)
                result *= resolution;
            return result * Features;
        }
    }

    [BackwardDerivative(getBwd)]
    FeatureVector get(StructuredBufferStorage<float> storage, int idx)
    {
        var result = FeatureVector();
        [ForceUnroll]
        for (int i = 0; i < Features; i++)
            result[i] = storage.read(offset + idx * Features + i);
        return result;
    }

    public void getBwd(DifferentialPtrPair<StructuredBufferStorage<float>> dstorage, int idx, FeatureVector.Differential doutput)
    {
        [ForceUnroll]
        for (int i = 0; i < Features; i++)
            dstorage.d.add(offset + idx * Features + i, doutput[i]);
    }

    // Helper function for scalar multiplication
    static FeatureVector scalarMul(FeatureVector v, float s)
    {
        var result = FeatureVector();
        [ForceUnroll]
        for (int i = 0; i < Features; i++)
            result[i] = v[i] * s;
        return result;
    }
    
    // Helper function for dot product
    static float dot(FeatureVector a, FeatureVector b)
    {
        float result = 0.0;
        [ForceUnroll]
        for (int i = 0; i < Features; i++)
            result += a[i] * b[i];
        return result;
    }

    [NoDiffThis, BackwardDerivative(sampleLinearBwd)]
    public FeatureVector sampleLinear(StructuredBufferStorage<float> storage, InlineVector<float, Dimension> p)
    {
        static_assert(Dimension == 1, "Linear interpolation is only supported for 1D inputs");
        
        let snapped = snapToLattice<InlineVector<float, Dimension>, float, Dimension>(p, resolution);
        
        let value0 = get(storage, snapped.lower[0]);
        let value1 = get(storage, snapped.upper[0]);
        
        let value = FeatureVector.interpolate(value0, value1, snapped.t[0]);
        
        return value;
    }

    [NoDiffThis, BackwardDerivative(sampleBilinearBwd)]
    public FeatureVector sampleBilinear(StructuredBufferStorage<float> storage, InlineVector<float, Dimension> p)
    {
        static_assert(Dimension == 2, "Bilinear interpolation is only supported for 2D inputs");
        
        let snapped = snapToLattice<InlineVector<float, Dimension>, float, Dimension>(p, resolution);
        
        let idx00 = snapped.lower[1] * resolution + snapped.lower[0];
        let idx01 = snapped.upper[1] * resolution + snapped.lower[0];
        let idx10 = snapped.lower[1] * resolution + snapped.upper[0];
        let idx11 = snapped.upper[1] * resolution + snapped.upper[0];
        
        let value00 = get(storage, idx00);
        let value01 = get(storage, idx01);
        let value10 = get(storage, idx10);
        let value11 = get(storage, idx11);
        
        let value0 = FeatureVector.interpolate(value00, value10, snapped.t[0]);
        let value1 = FeatureVector.interpolate(value01, value11, snapped.t[0]);
        
        let value = FeatureVector.interpolate(value0, value1, snapped.t[1]);
        
        return value;
    }

    [NoDiffThis, BackwardDerivative(sampleTrilinearBwd)]
    public FeatureVector sampleTrilinear(StructuredBufferStorage<float> storage, InlineVector<float, Dimension> p)
    {
        let snapped = snapToLattice<InlineVector<float, Dimension>, float, Dimension>(p, resolution);

        let value000 = get(storage, snapped.lower[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.lower[0]);
        let value001 = get(storage, snapped.upper[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.lower[0]);
        let value010 = get(storage, snapped.lower[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.lower[0]);
        let value011 = get(storage, snapped.upper[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.lower[0]);
        let value100 = get(storage, snapped.lower[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.upper[0]);
        let value101 = get(storage, snapped.upper[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.upper[0]);
        let value110 = get(storage, snapped.lower[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.upper[0]);
        let value111 = get(storage, snapped.upper[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.upper[0]);
        
        let value00 = FeatureVector.interpolate(value000, value100, snapped.t[0]);
        let value01 = FeatureVector.interpolate(value001, value101, snapped.t[0]);
        let value10 = FeatureVector.interpolate(value010, value110, snapped.t[0]);
        let value11 = FeatureVector.interpolate(value011, value111, snapped.t[0]);
        
        let value0 = FeatureVector.interpolate(value00, value10, snapped.t[1]);
        let value1 = FeatureVector.interpolate(value01, value11, snapped.t[1]);
        
        let value = FeatureVector.interpolate(value0, value1, snapped.t[2]);
        
        return value;
    }

    public void sampleLinearBwd(
        DifferentialPtrPair<StructuredBufferStorage<float>> dstorage,
        inout DifferentialPair<InlineVector<float, Dimension>> dp,
        FeatureVector.Differential doutput
    )
    {
        static_assert(Dimension == 1, "Linear interpolation is only supported for 1D inputs");
        
        // Reconstruct the snapped lattice information
        let snapped = snapToLattice<InlineVector<float, Dimension>, float, Dimension>(dp.p, resolution);
        
        // Get the corner values for gradient computation
        let value0 = get(dstorage.p, snapped.lower[0]);
        let value1 = get(dstorage.p, snapped.upper[0]);
        
        // Backward pass: propagate gradients through linear interpolation
        // Linear interpolation: value = value0 * (1-t[0]) + value1 * t[0]
        let dvalue0 = scalarMul(doutput, 1.0 - snapped.t[0]);
        let dvalue1 = scalarMul(doutput, snapped.t[0]);
        
        // Accumulate gradients to storage
        getBwd(dstorage, snapped.lower[0], dvalue0);
        getBwd(dstorage, snapped.upper[0], dvalue1);
        
        // Compute gradients with respect to input position
        // The derivative of snapped.t[0] with respect to p[0] is resolution
        let dt_dp = float(resolution);
        
        // Compute gradient contribution for the single dimension
        float grad_d = dot(doutput, value1.sub(value0)) * dt_dp;
        
        var dp_d = dp.d;
        dp_d[0] += grad_d;
        dp = DifferentialPair<InlineVector<float, Dimension>>(dp.p, dp_d);
    }

    public void sampleBilinearBwd(
        DifferentialPtrPair<StructuredBufferStorage<float>> dstorage,
        inout DifferentialPair<InlineVector<float, Dimension>> dp,
        FeatureVector.Differential doutput
    )
    {
        static_assert(Dimension == 2, "Bilinear interpolation is only supported for 2D inputs");
        
        // Reconstruct the snapped lattice information
        let snapped = snapToLattice<InlineVector<float, Dimension>, float, Dimension>(dp.p, resolution);
        
        // Calculate indices for the 4 corners
        let idx00 = snapped.lower[1] * resolution + snapped.lower[0];
        let idx01 = snapped.upper[1] * resolution + snapped.lower[0];
        let idx10 = snapped.lower[1] * resolution + snapped.upper[0];
        let idx11 = snapped.upper[1] * resolution + snapped.upper[0];
        
        // Get the corner values for gradient computation
        let value00 = get(dstorage.p, idx00);
        let value01 = get(dstorage.p, idx01);
        let value10 = get(dstorage.p, idx10);
        let value11 = get(dstorage.p, idx11);
        
        // Forward pass interpolation (reconstructed for gradient computation)
        let value0 = FeatureVector.interpolate(value00, value10, snapped.t[0]);
        let value1 = FeatureVector.interpolate(value01, value11, snapped.t[0]);
        
        // Backward pass: propagate gradients through interpolation chain
        // Final interpolation: value = value0 * (1-t[1]) + value1 * t[1]
        let dvalue0 = scalarMul(doutput, 1.0 - snapped.t[1]);
        let dvalue1 = scalarMul(doutput, snapped.t[1]);
        
        // First level interpolation: value0 = value00 * (1-t[0]) + value10 * t[0]
        // value1 = value01 * (1-t[0]) + value11 * t[0]
        let dvalue00 = scalarMul(dvalue0, 1.0 - snapped.t[0]);
        let dvalue10 = scalarMul(dvalue0, snapped.t[0]);
        let dvalue01 = scalarMul(dvalue1, 1.0 - snapped.t[0]);
        let dvalue11 = scalarMul(dvalue1, snapped.t[0]);
        
        // Accumulate gradients to storage
        getBwd(dstorage, idx00, dvalue00);
        getBwd(dstorage, idx01, dvalue01);
        getBwd(dstorage, idx10, dvalue10);
        getBwd(dstorage, idx11, dvalue11);
        
        // Compute gradients with respect to input position
        // For each dimension, we need to compute the derivative of the interpolation weights
        [ForceUnroll]
        for (int d = 0; d < Dimension; d++)
        {
            // The derivative of snapped.t[d] with respect to p[d] is resolution
            let dt_dp = float(resolution);
            
            // Compute gradient contribution for this dimension
            float grad_d = 0.0;
            
            if (d == 0)
            {
                // Derivatives with respect to t[0]
                grad_d += dot(dvalue0, value10.sub(value00)) * dt_dp;
                grad_d += dot(dvalue1, value11.sub(value01)) * dt_dp;
            }
            else if (d == 1)
            {
                // Derivatives with respect to t[1]
                grad_d += dot(doutput, value1.sub(value0)) * dt_dp;
            }
            
            var dp_d = dp.d;
            dp_d[d] += grad_d;
            dp = DifferentialPair<InlineVector<float, Dimension>>(dp.p, dp_d);
        }
    }

    public void sampleTrilinearBwd(
        DifferentialPtrPair<StructuredBufferStorage<float>> dstorage,
        inout DifferentialPair<InlineVector<float, Dimension>> dp,
        FeatureVector.Differential doutput
    )
    {
        // Reconstruct the snapped lattice information
        let snapped = snapToLattice<InlineVector<float, Dimension>, float, Dimension>(dp.p, resolution);
        
        // Calculate indices for the 8 corners
        let idx000 = snapped.lower[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.lower[0];
        let idx001 = snapped.upper[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.lower[0];
        let idx010 = snapped.lower[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.lower[0];
        let idx011 = snapped.upper[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.lower[0];
        let idx100 = snapped.lower[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.upper[0];
        let idx101 = snapped.upper[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.upper[0];
        let idx110 = snapped.lower[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.upper[0];
        let idx111 = snapped.upper[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.upper[0];
        
        // Get the corner values for gradient computation
        let value000 = get(dstorage.p, idx000);
        let value001 = get(dstorage.p, idx001);
        let value010 = get(dstorage.p, idx010);
        let value011 = get(dstorage.p, idx011);
        let value100 = get(dstorage.p, idx100);
        let value101 = get(dstorage.p, idx101);
        let value110 = get(dstorage.p, idx110);
        let value111 = get(dstorage.p, idx111);
        
        // Forward pass interpolation (reconstructed for gradient computation)
        let value00 = FeatureVector.interpolate(value000, value100, snapped.t[0]);
        let value01 = FeatureVector.interpolate(value001, value101, snapped.t[0]);
        let value10 = FeatureVector.interpolate(value010, value110, snapped.t[0]);
        let value11 = FeatureVector.interpolate(value011, value111, snapped.t[0]);
        
        let value0 = FeatureVector.interpolate(value00, value10, snapped.t[1]);
        let value1 = FeatureVector.interpolate(value01, value11, snapped.t[1]);
        
        // Backward pass: propagate gradients through interpolation chain
        // Final interpolation: value = value0 * (1-t[2]) + value1 * t[2]
        let dvalue0 = scalarMul(doutput, 1.0 - snapped.t[2]);
        let dvalue1 = scalarMul(doutput, snapped.t[2]);
        
        // Second level interpolation: value0 = value00 * (1-t[1]) + value10 * t[1]
        // value1 = value01 * (1-t[1]) + value11 * t[1]
        let dvalue00 = scalarMul(dvalue0, 1.0 - snapped.t[1]);
        let dvalue10 = scalarMul(dvalue0, snapped.t[1]);
        let dvalue01 = scalarMul(dvalue1, 1.0 - snapped.t[1]);
        let dvalue11 = scalarMul(dvalue1, snapped.t[1]);
        
        // First level interpolation: value00 = value000 * (1-t[0]) + value100 * t[0]
        // value01 = value001 * (1-t[0]) + value101 * t[0]
        // value10 = value010 * (1-t[0]) + value110 * t[0]
        // value11 = value011 * (1-t[0]) + value111 * t[0]
        let dvalue000 = scalarMul(dvalue00, 1.0 - snapped.t[0]);
        let dvalue100 = scalarMul(dvalue00, snapped.t[0]);
        let dvalue001 = scalarMul(dvalue01, 1.0 - snapped.t[0]);
        let dvalue101 = scalarMul(dvalue01, snapped.t[0]);
        let dvalue010 = scalarMul(dvalue10, 1.0 - snapped.t[0]);
        let dvalue110 = scalarMul(dvalue10, snapped.t[0]);
        let dvalue011 = scalarMul(dvalue11, 1.0 - snapped.t[0]);
        let dvalue111 = scalarMul(dvalue11, snapped.t[0]);
        
        // Accumulate gradients to storage
        getBwd(dstorage, idx000, dvalue000);
        getBwd(dstorage, idx001, dvalue001);
        getBwd(dstorage, idx010, dvalue010);
        getBwd(dstorage, idx011, dvalue011);
        getBwd(dstorage, idx100, dvalue100);
        getBwd(dstorage, idx101, dvalue101);
        getBwd(dstorage, idx110, dvalue110);
        getBwd(dstorage, idx111, dvalue111);
        
        // Compute gradients with respect to input position
        // For each dimension, we need to compute the derivative of the interpolation weights
        [ForceUnroll]
        for (int d = 0; d < Dimension; d++)
        {
            // The derivative of snapped.t[d] with respect to p[d] is resolution
            let dt_dp = float(resolution);
            
            // Compute gradient contribution for this dimension
            float grad_d = 0.0;
            
            if (d == 0)
            {
                // Derivatives with respect to t[0]
                grad_d += dot(dvalue00, value100.sub(value000)) * dt_dp;
                grad_d += dot(dvalue01, value101.sub(value001)) * dt_dp;
                grad_d += dot(dvalue10, value110.sub(value010)) * dt_dp;
                grad_d += dot(dvalue11, value111.sub(value011)) * dt_dp;
            }
            else if (d == 1)
            {
                // Derivatives with respect to t[1]
                grad_d += dot(dvalue0, value10.sub(value00)) * dt_dp;
                grad_d += dot(dvalue1, value11.sub(value01)) * dt_dp;
            }
            else if (d == 2)
            {
                // Derivatives with respect to t[2]
                grad_d += dot(doutput, value1.sub(value0)) * dt_dp;
            }
            
            var dp_d = dp.d;
            dp_d[d] += grad_d;
            dp = DifferentialPair<InlineVector<float, Dimension>>(dp.p, dp_d);
        }
    }
    
    // Only for derived structures
    public void sampleBwd<OutputVector>(
        DifferentialPtrPair<StructuredBufferStorage<float>> dstorage,
        inout DifferentialPair<InlineVector<float, Dimension>> dp,
        OutputVector.Differential doutput
    )
        where OutputVector : IVector<float, Features>
    {
        if (Dimension == 1)
            return sampleLinearBwd(dstorage, dp, __slang_noop_cast<FeatureVector>(doutput));
        else if (Dimension == 2)
            return sampleBilinearBwd(dstorage, dp, __slang_noop_cast<FeatureVector>(doutput));
        else if (Dimension == 3)
            return sampleTrilinearBwd(dstorage, dp, __slang_noop_cast<FeatureVector>(doutput));
        else
            static_assert(false, "DenseGridImpl only supports 1D, 2D, and 3D inputs");
    }

    [NoDiffThis, BackwardDifferentiable]
    public FeatureVector sample(StructuredBufferStorage<float> storage, InlineVector<float, Dimension> p)
    {
        if (Dimension == 1)
            return sampleLinear(storage, p);
        else if (Dimension == 2)
            return sampleBilinear(storage, p);
        else if (Dimension == 3)
            return sampleTrilinear(storage, p);
        else
            static_assert(false, "DenseGridImpl only supports 1D, 2D, and 3D inputs");

        return FeatureVector();
    }
}

// TODO: some interface that is more general and mlp/encoder... this has the same structure...
public struct DenseGrid<int Dimension, int Features>
{
    public typealias Impl = DenseGridImpl<Dimension, Features>;
    public typealias FeatureVector = InlineVector<float, Features>;

    public typealias Parameters = StructuredBufferStorage<float>;
    public typealias Configuration = This;
    public typealias Dual = DifferentialPtrPair<Parameters>;

    RWStructuredBuffer<float> parameterBuffer;
    RWStructuredBuffer<float> gradientBuffer;

    int offset;
    int resolution;

    public property Parameters parameters
    {
        get {
            return StructuredBufferStorage<float>(parameterBuffer);
        }
    }

    // TODO: temporary
    public property StructuredBufferStorage<float> gradients
    {
        get {
            return StructuredBufferStorage<float>(gradientBuffer);
        }
    }

    public property Configuration configuration
    {
        get {
            return this;
        }
    }

    public property Dual dual
    {
        get {
            let p = StructuredBufferStorage<float>(parameterBuffer);
            let g = StructuredBufferStorage<float>(gradientBuffer);
            return DifferentialPtrPair<StructuredBufferStorage<float>>(p, g);
        }
    }
    
    public void update<Optimizer>(Optimizer optimizer, inout Optimizer.State state, uint tid)
        where Optimizer : IOptimizer<float>
    {
        optimizer.step(state, parameterBuffer[tid], gradientBuffer[tid]);
        gradientBuffer[tid] = 0.0;
    }

    [BackwardDifferentiable]
    public static FeatureVector eval(Parameters parameters, no_diff Configuration configuration, InlineVector<float, Dimension> p)
    {
        return Impl(
            configuration.offset,
            configuration.resolution
        ) .sample(parameters, p);
    }
}