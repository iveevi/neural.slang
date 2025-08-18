module shapes;

import camera;

public interface Shape
{
    public Optional<RayIntersection> intersection(Ray ray);
}

public struct SphericalCoordinates
{
    // [0, 2pi]
    public float theta;
    // [-pi/2, pi/2]
    public float phi;

    float2 normalized()
    {
        let ntheta = theta / (2.0 * float.getPi());
        let nphi = (phi + 0.5 * float.getPi()) / float.getPi();
        return float2(ntheta, nphi);
    }

    static SphericalCoordinates from(float3 d)
    {
        let theta = atan2(d.x, d.y);
        let phi = asin(d.z);
        return SphericalCoordinates(theta, phi);
    }
}

public struct Sphere : Shape
{
    public float3 center;
    public float radius;

    public Optional<RayIntersection> intersection(Ray ray)
    {
        float3 oc = ray.origin - center;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0 * dot(oc, ray.direction);
        float c = dot(oc, oc) - radius * radius;
        
        float discriminant = b * b - 4.0 * a * c;
        if (discriminant < 0.0)
            return {};
        
        float sqrtDiscriminant = sqrt(discriminant);
        float t1 = (-b - sqrtDiscriminant) / (2.0 * a);
        float t2 = (-b + sqrtDiscriminant) / (2.0 * a);
        
        float t = (t1 > 0.0) ? t1 : t2;
        if (t <= 0.0)
            return {};
        
        float3 point = ray.at(t);
        float3 normal = normalize(point - center);
        
        return RayIntersection(t, point, normal);
    }
}

public struct Box : Shape
{
    public float3 low;
    public float3 high;

    [NoDiffThis, Differentiable]
    public float3 normalized(float3 point)
    {
        return (point - low) / (high - low);
    }

    public bool contains(float3 point)
    {
        return all(point >= low && point <= high);
    }

    public Optional<RayIntersection> intersection(Ray ray)
    {
        float3 dir = ray.direction;
        float3 invDir = 1.0 / dir;
        
        float tMin = (low.x - ray.origin.x) * invDir.x;
        float tMax = (high.x - ray.origin.x) * invDir.x;
        
        if (tMin > tMax)
        {
            float temp = tMin;
            tMin = tMax;
            tMax = temp;
        }
        
        float tyMin = (low.y - ray.origin.y) * invDir.y;
        float tyMax = (high.y - ray.origin.y) * invDir.y;
        
        if (tyMin > tyMax)
        {
            float temp = tyMin;
            tyMin = tyMax;
            tyMax = temp;
        }
        
        if (tMin > tyMax || tyMin > tMax)
            return {};
            
        tMin = max(tMin, tyMin);
        tMax = min(tMax, tyMax);
        
        float tzMin = (low.z - ray.origin.z) * invDir.z;
        float tzMax = (high.z - ray.origin.z) * invDir.z;
        
        if (tzMin > tzMax)
        {
            float temp = tzMin;
            tzMin = tzMax;
            tzMax = temp;
        }
        
        if (tMin > tzMax || tzMin > tMax)
            return {};
            
        tMin = max(tMin, tzMin);
        tMax = min(tMax, tzMax);
        
        if (tMax < 0.0)
            return {};
            
        float t = tMin >= 0.0 ? tMin : tMax;
        if (t < 0.0)
            return {};
        
        float3 point = ray.at(t);
        float3 normal = float3(0.0);
        
        float epsilon = 1e-6;
        if (abs(point.x - low.x) < epsilon)
            normal = float3(-1.0, 0.0, 0.0);
        else if (abs(point.x - high.x) < epsilon)
            normal = float3(1.0, 0.0, 0.0);
        else if (abs(point.y - low.y) < epsilon)
            normal = float3(0.0, -1.0, 0.0);
        else if (abs(point.y - high.y) < epsilon)
            normal = float3(0.0, 1.0, 0.0);
        else if (abs(point.z - low.z) < epsilon)
            normal = float3(0.0, 0.0, -1.0);
        else if (abs(point.z - high.z) < epsilon)
            normal = float3(0.0, 0.0, 1.0);
        
        return RayIntersection(t, point, normal);
    }

    public static Box fromCenterAndSize(float3 center, float3 size)
    {
        return Box(center - size * 0.5, center + size * 0.5);
    }
}

public struct CylinderCoordinates
{
    // [0, 1]
    public float r;
    // [-pi, pi]
    public float theta;
    // [-1, 1]
    public float y;

    public float normalizedTheta()
    {
        return (theta + float.getPi()) / (2.0 * float.getPi());
    }
}

public struct Cylinder : Shape
{
    public float3 center;
    public float radius;
    public float height;

    public Optional<RayIntersection> intersection(Ray ray)
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
        
        float3 normal = normalize(float3(relativeHit.x, 0.0, relativeHit.z));
        
        return RayIntersection(t, hitPoint, normal);
    }

    public CylinderCoordinates toCylindrical(float3 point)
    {
        float3 relative = point - center;
        float r = length(relative.xz) / radius;
        float theta = atan2(relative.z, relative.x);
        float y = (relative.y + height * 0.5) / height;
        return CylinderCoordinates(r, theta, y);
    }
}

public struct Triangle : Shape
{
    public float3 v0;
    public float3 v1;
    public float3 v2;

    public Optional<RayIntersection> intersection(Ray ray)
    {
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        float3 h = cross(ray.direction, edge2);
        float a = dot(edge1, h);
        
        if (abs(a) < 1e-6)
            return {};
        
        float f = 1.0 / a;
        float3 s = ray.origin - v0;
        float u = f * dot(s, h);
        
        if (u < 0.0 || u > 1.0)
            return {};
        
        float3 q = cross(s, edge1);
        float v = f * dot(ray.direction, q);
        
        if (v < 0.0 || u + v > 1.0)
            return {};
        
        float t = f * dot(edge2, q);
        
        if (t <= 0.0)
            return {};
        
        float3 point = ray.at(t);
        float3 normal = normalize(cross(edge1, edge2));
        
        return RayIntersection(t, point, normal);
    }
}