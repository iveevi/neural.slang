module shapes;

import camera;

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

public struct Cylinder
{
    public float3 center;
    public float radius;
    public float height;

    public Optional<float3> intersection(Ray ray)
    {
        float3 relativeOrigin = ray.origin - center;
        float3 dir = ray.direction;
        
        float a = dir.x * dir.x + dir.z * dir.z;
        float b = 2.0 * (relativeOrigin.x * dir.x + relativeOrigin.z * dir.z);
        float c = relativeOrigin.x * relativeOrigin.x + relativeOrigin.z * relativeOrigin.z - radius * radius;
        
        float discriminant = b * b - 4.0 * a * c;
        if (discriminant < 0.0 || a == 0.0)
            return {};
        
        float sqrtDiscriminant = sqrt(discriminant);
        float t1 = (-b - sqrtDiscriminant) / (2.0 * a);
        float t2 = (-b + sqrtDiscriminant) / (2.0 * a);
        
        float t = (t1 > 0.0) ? t1 : t2;
        if (t <= 0.0)
            return {};
        
        float3 hitPoint = ray.origin + t * dir;
        float3 relativeHit = hitPoint - center;

        if (abs(relativeHit.y) > height * 0.5)
            return {};
        
        return hitPoint;
    }

    public float3 toCylindrical(float3 point)
    {
        float3 relative = point - center;
        float r = length(relative.xz) / radius;
        float theta = atan2(relative.z, relative.x);
        float y = (relative.y + height * 0.5) / height;
        return float3(r, theta, y);
    }
}

public struct Triangle
{
    public float3 v0;
    public float3 v1;
    public float3 v2;
}