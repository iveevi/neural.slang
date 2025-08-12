module camera;

public struct Ray
{
    public float3 origin;
    public float3 direction;
};

public struct RayFrame
{
    public float3 origin;
    public float3 lower_left;
    public float3 horizontal;
    public float3 vertical;

    public Ray operator()(float u, float v)
    {
        float3 dir = normalize(lower_left + u * horizontal + v * vertical - origin);
        return Ray(origin, dir);
    }
};