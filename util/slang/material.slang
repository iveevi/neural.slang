module brdf;

struct SurfaceFrame
{
    float3 position;
    float3 normal;
    // TODO: random numbers...
}

interface IMaterial
{
    float3 eval(SurfaceFrame frame, float3 wi, out float3 wo, out float pdf);
}

// Essential materials
struct LambertDiffuse : IMaterial
{
    float3 eval(SurfaceFrame frame, float3 wi, out float3 wo, out float pdf)
    {
        wo = -wi;
        pdf = 1.0;
        return float3(1.0) / float.getPi();
    }
}