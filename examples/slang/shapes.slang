module shapes;

public struct Sphere
{
    public float3 center;
    public float radius;
}

public struct Box
{
    public float3 low;
    public float3 high;

    public static Box fromCenterAndSize(float3 center, float3 size)
    {
        return Box(center - size * 0.5, center + size * 0.5);
    }
}

public struct Triangle
{
    public float3 v0;
    public float3 v1;
    public float3 v2;
}