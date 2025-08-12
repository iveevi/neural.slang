module sdf;

import mesh;
import camera;
import shapes;

// Helper function for dot product squared
float dot2(float3 v) { return dot(v, v); }

public interface ISignedDistanceField
{
    public float sdf(float3 p);
}

// Essential primitives
public extension Sphere : ISignedDistanceField
{
    public float sdf(float3 p)
    {
        return length(p - center) - radius;
    }
}

public extension Box : ISignedDistanceField
{
    public float sdf(float3 p)
    {
        return max(max(p.x - high.x, low.x - p.x), max(max(p.y - high.y, low.y - p.y), max(p.z - high.z, low.z - p.z)));
    }
}

public extension Triangle : ISignedDistanceField
{
    public float sdf(float3 p)
    {
        // Port of Inigo Quilez's udTriangle function
        float3 ba = v1 - v0; float3 pa = p - v0;
        float3 cb = v2 - v1; float3 pb = p - v1;
        float3 ac = v0 - v2; float3 pc = p - v2;
        float3 nor = cross(ba, ac);

        return sqrt(
            (sign(dot(cross(ba, nor), pa)) +
             sign(dot(cross(cb, nor), pb)) +
             sign(dot(cross(ac, nor), pc)) < 2.0)
             ?
             min(min(
                 dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
                 dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
                 dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc))
             :
             dot(nor, pa) * dot(nor, pa) / dot2(nor));
    }
}

public extension TriangleMesh : ISignedDistanceField
{
    public float sdf(float3 p)
    {
        float min_d = 1e10;

        for (int i = 0; i < triangles.getCount(); i++)
        {
            uint3 idx = triangles[i];
            let tri = Triangle(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
            float d = tri.sdf(p);
            if (d < min_d)
                min_d = d;
        }

        return min_d;
    }
}

// Intersection routines
public interface ISphereTraceIntersection
{
    public static This sphereTrace<Object>(Object object, Ray ray, int maxIters, float tMin, float tMax)
        where Object : ISignedDistanceField;
}

public struct PositionalSphereTrace : ISphereTraceIntersection
{
    public float t;
    public float3 p;

    public static This sphereTrace<Object>(Object object, Ray ray, int maxIters, float tMin, float tMax)
        where Object : ISignedDistanceField
    {
        float t = 0.0;
        for (int i = 0; i < maxIters; i++)
        {
            float3 p = ray.origin + t * ray.direction;
            float d = object.sdf(p);
            if (d < 0.001 || t > tMax)
                return This(t, p);
            if (t < tMin)
                return This(0.0, float3(0.0));
            t += d;
        }

        return This(0.0, float3(0.0));
    }
}

public struct PositionNormalSphereTrace : ISphereTraceIntersection
{
    public float t;
    public float3 p;
    public float3 n;

    public static This sphereTrace<Object>(Object object, Ray ray, int maxIters, float tMin, float tMax)
        where Object : ISignedDistanceField
    {
        float t = 0.0;
        for (int i = 0; i < maxIters; i++)
        {
            float3 p = ray.origin + t * ray.direction;
            float d = object.sdf(p);
            if (d < 0.001 || t > tMax)
            {
                // Calculate gradient
                // TODO: use autodiff for this

                float3 px_plus_eps = p + float3(1e-4, 0.0, 0.0);
                float3 px_minus_eps = p - float3(1e-4, 0.0, 0.0);
                float3 py_plus_eps = p + float3(0.0, 1e-4, 0.0);
                float3 py_minus_eps = p - float3(0.0, 1e-4, 0.0);
                float3 pz_plus_eps = p + float3(0.0, 0.0, 1e-4);
                float3 pz_minus_eps = p - float3(0.0, 0.0, 1e-4);

                float dx = object.sdf(px_plus_eps) - object.sdf(px_minus_eps);
                float dy = object.sdf(py_plus_eps) - object.sdf(py_minus_eps);
                float dz = object.sdf(pz_plus_eps) - object.sdf(pz_minus_eps);

                float3 n = normalize(float3(dx, dy, dz));
                
                return This(t, p, n);
            }
            if (t < tMin)
                return This(0.0, float3(0.0), float3(0.0));
            t += d;
        }

        return This(0.0, float3(0.0), float3(0.0));
    }
}