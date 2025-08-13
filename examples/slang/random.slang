module random;

public struct PCG
{
	public static const uint64_t PCG_MULTIPLIER = 0x5851f42d4c957f2d;
	public static const uint64_t PCG_INCREMENT = 0x14057b7ef767814f;

    uint64_t state;
    
    public __init(uint seed)
    {
        this.state = uint64_t(seed);
        this.nextUint();
    }
    
    [mutating]
    public uint nextUint()
    {
        uint64_t oldstate = this.state;
        this.state = oldstate * PCG_MULTIPLIER + PCG_INCREMENT;
        uint xorshifted = uint((oldstate >> 18) ^ oldstate) >> 27;
        uint rot = uint(oldstate >> 59);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1) & 31));
    }
    
    [mutating]
    public float rand()
    {
        uint randomBits = this.nextUint();
        return float(randomBits >> 8) * (1.0f / 16777216.0f);
    }
    
    [mutating]
    public vector<float, N> rand<int N>()
    {
        vector<float, N> result;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result[i] = this.rand();
        return result;
    }
}