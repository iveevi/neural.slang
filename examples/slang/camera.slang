module camera;

public struct Ray
{
    public float3 origin;
    public float3 direction;

    [NoDiffThis, Differentiable]
    public float3 at(float t)
    {
        return origin + direction * t;
    }
};

public struct RayFrame
{
    public float3 origin;
    public float3 lower_left;
    public float3 horizontal;
    public float3 vertical;

    [NoDiffThis, Differentiable]
    public Ray rayAt(float2 uv)
    {
        float3 dir = normalize(lower_left + uv.x * horizontal + uv.y * vertical - origin);
        return Ray(origin, dir);
    }
};