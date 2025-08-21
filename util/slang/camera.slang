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

public struct RayIntersection
{
    public float t;
    public float3 point;
    public float3 normal;

    bool valid()
    {
        return t >= 0.0;
    }
}

public struct CoordinateFrame
{
    public float3 normal;
    public float3 tangent;
    public float3 bitangent;

    public __init(float3 normal, float3 tangent, float3 bitangent)
    {
        this.normal = normal;
        this.tangent = tangent;
        this.bitangent = bitangent;
    }

    public float3 toLocal(float3 point)
    {
        return float3(
            dot(point, tangent),
            dot(point, bitangent),
            dot(point, normal)
        );
    }
}

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