module mesh;

public struct TriangleMesh
{
    // TODO: make this bindless
    public RWStructuredBuffer<float[3]> vertices;
    public RWStructuredBuffer<uint[3]> triangles;
    public uint triangleCount;
};
