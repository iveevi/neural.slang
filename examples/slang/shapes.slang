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
}

public struct Triangle
{
    public float3 v0;
    public float3 v1;
    public float3 v2;
}