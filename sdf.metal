#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 3 "examples/slang/camera.slang"
struct Ray_0
{
    float3 origin_0;
    float3 direction_0;
};


#line 3
Ray_0 Ray_x24init_0(const float3 thread* origin_1, const float3 thread* direction_1)
{

#line 3
    thread Ray_0 _S1;

    (&_S1)->origin_0 = *origin_1;
    (&_S1)->direction_0 = *direction_1;

#line 3
    return _S1;
}




struct RayFrame_0
{
    float3 origin_2;
    float3 lower_left_0;
    float3 horizontal_0;
    float3 vertical_0;
};


#line 16
Ray_0 RayFrame_rayAt_0(const RayFrame_0 thread* this_0, const float2 thread* uv_0)
{
    float3 dir_0 = normalize(this_0->lower_left_0 + float3((*uv_0).x)  * this_0->horizontal_0 + float3((*uv_0).y)  * this_0->vertical_0 - this_0->origin_2);

#line 18
    float3 _S2 = this_0->origin_2;

#line 18
    float3 _S3 = dir_0;

#line 18
    Ray_0 _S4 = Ray_x24init_0(&_S2, &_S3);
    return _S4;
}


#line 82 "examples/slang/sdf.slang"
struct PositionalSphereTrace_0
{
    float t_0;
    float3 p_0;
};


#line 82
PositionalSphereTrace_0 PositionalSphereTrace_x24init_0(float t_1, const float3 thread* p_1)
{

#line 82
    thread PositionalSphereTrace_0 _S5;

    (&_S5)->t_0 = t_1;
    (&_S5)->p_0 = *p_1;

#line 82
    return _S5;
}


#line 23085 "hlsl.meta.slang"
int getCount_0(uint3 device* this_1)
{

#line 23085
    uint _elementCount_0;
    uint _stride_0;
    this_1.GetDimensions(_elementCount_0, _stride_0);
    uint2 _S6 = uint2(_elementCount_0, _stride_0);

#line 23085
    return int(_S6.x);
}


#line 15 "examples/slang/shapes.slang"
struct Triangle_0
{
    float3 v0_0;
    float3 v1_0;
    float3 v2_0;
};


#line 15
Triangle_0 Triangle_x24init_0(const float3 thread* v0_1, const float3 thread* v1_1, const float3 thread* v2_1)
{

#line 15
    thread Triangle_0 _S7;

    (&_S7)->v0_0 = *v0_1;
    (&_S7)->v1_0 = *v1_1;
    (&_S7)->v2_0 = *v2_1;

#line 15
    return _S7;
}


#line 8 "examples/slang/sdf.slang"
float dot2_0(const float3 thread* v_0)
{

#line 8
    return dot(*v_0, *v_0);
}


#line 34
float sdf_0(const Triangle_0 thread* this_2, const float3 thread* p_2)
{

    float3 ba_0 = this_2->v1_0 - this_2->v0_0;

#line 37
    float3 pa_0 = *p_2 - this_2->v0_0;
    float3 cb_0 = this_2->v2_0 - this_2->v1_0;

#line 38
    float3 pb_0 = *p_2 - this_2->v1_0;
    float3 ac_0 = this_2->v0_0 - this_2->v2_0;

#line 39
    float3 pc_0 = *p_2 - this_2->v2_0;
    float3 nor_0 = cross(ba_0, ac_0);

#line 40
    float _S8;

#line 46
    if(float((int(sign((dot(cross(ba_0, nor_0), pa_0))))) + (int(sign((dot(cross(cb_0, nor_0), pb_0))))) + (int(sign((dot(cross(ac_0, nor_0), pc_0)))))) < 2.0)
    {
        float _S9 = dot(ba_0, pa_0);

#line 48
        float3 _S10 = ba_0;

#line 48
        float _S11 = dot2_0(&_S10);

#line 48
        float3 _S12 = ba_0 * float3(clamp(_S9 / _S11, 0.0, 1.0))  - pa_0;

#line 48
        float _S13 = dot2_0(&_S12);
        float _S14 = dot(cb_0, pb_0);

#line 49
        float3 _S15 = cb_0;

#line 49
        float _S16 = dot2_0(&_S15);

#line 49
        float3 _S17 = cb_0 * float3(clamp(_S14 / _S16, 0.0, 1.0))  - pb_0;

#line 49
        float _S18 = dot2_0(&_S17);

#line 47
        float _S19 = min(_S13, _S18);


        float _S20 = dot(ac_0, pc_0);

#line 50
        float3 _S21 = ac_0;

#line 50
        float _S22 = dot2_0(&_S21);

#line 50
        float3 _S23 = ac_0 * float3(clamp(_S20 / _S22, 0.0, 1.0))  - pc_0;

#line 50
        float _S24 = dot2_0(&_S23);

#line 50
        _S8 = min(_S19, _S24);

#line 46
    }
    else
    {



        float _S25 = dot(nor_0, pa_0);

#line 52
        float _S26 = _S25 * _S25;

#line 52
        float3 _S27 = nor_0;

#line 52
        float _S28 = dot2_0(&_S27);

#line 52
        _S8 = _S26 / _S28;

#line 46
    }

#line 42
    return sqrt(_S8);
}


#line 58
float sdf_1(float3 device* this_vertices_0, uint3 device* this_triangles_0, const float3 thread* p_3)
{

#line 58
    float min_d_0 = 1.0e+10;

#line 58
    int i_0 = int(0);



    for(;;)
    {

#line 62
        int _S29 = getCount_0(this_triangles_0);

#line 62
        if(i_0 < _S29)
        {
        }
        else
        {

#line 62
            break;
        }
        uint3 device* _S30 = this_triangles_0+i_0;
        float3 _S31 = *(this_vertices_0+(*_S30).y);

#line 65
        float3 _S32 = *(this_vertices_0+(*_S30).z);

#line 65
        float3 _S33 = *(this_vertices_0+(*_S30).x);

#line 65
        float3 _S34 = _S31;

#line 65
        float3 _S35 = _S32;

#line 65
        Triangle_0 _S36 = Triangle_x24init_0(&_S33, &_S34, &_S35);

#line 65
        Triangle_0 _S37 = _S36;

#line 65
        float3 _S38 = *p_3;

#line 65
        float _S39 = sdf_0(&_S37, &_S38);

        if(_S39 < min_d_0)
        {

#line 67
            min_d_0 = _S39;

#line 67
        }

#line 62
        i_0 = i_0 + int(1);

#line 62
    }

#line 71
    return min_d_0;
}


#line 87
PositionalSphereTrace_0 PositionalSphereTrace_sphereTrace_0(float3 device* object_vertices_0, uint3 device* object_triangles_0, const Ray_0 thread* ray_0, int maxIters_0, float tMin_0, float tMax_0)
{

#line 87
    int i_1 = int(0);

#line 87
    float t_2 = tMin_0;



    for(;;)
    {

#line 91
        if(i_1 < maxIters_0)
        {
        }
        else
        {

#line 91
            break;
        }
        float3 p_4 = ray_0->origin_0 + float3(t_2)  * ray_0->direction_0;

#line 93
        float3 _S40 = p_4;

#line 93
        float _S41 = sdf_1(object_vertices_0, object_triangles_0, &_S40);

        if((abs(_S41)) < 0.00100000004749745)
        {

#line 95
            float3 _S42 = p_4;

#line 95
            PositionalSphereTrace_0 _S43 = PositionalSphereTrace_x24init_0(t_2, &_S42);
            return _S43;
        }

#line 97
        if(t_2 > tMax_0)
        {

#line 97
            float3 _S44 = float3(0.0) ;

#line 97
            PositionalSphereTrace_0 _S45 = PositionalSphereTrace_x24init_0(0.0, &_S44);
            return _S45;
        }

#line 99
        float t_3 = t_2 + _S41;

#line 91
        i_1 = i_1 + int(1);

#line 91
        t_2 = t_3;

#line 91
    }

#line 91
    float3 _S46 = float3(0.0) ;

#line 91
    PositionalSphereTrace_0 _S47 = PositionalSphereTrace_x24init_0(0.0, &_S46);

#line 102
    return _S47;
}


#line 3 "examples/slang/mesh.slang"
struct TriangleMesh_0
{
    float3 device* vertices_0;
    uint3 device* triangles_0;
};


#line 4319 "hlsl.meta.slang"
struct GlobalParams_0
{
    RayFrame_0 rayFrame_0;
    uint2 targetResolution_0;
};


#line 4319
struct KernelContext_0
{
    TriangleMesh_0 constant* mesh_0;
    texture2d<float, access::read_write> targetTexture_0;
    GlobalParams_0 constant* globalParams_0;
};


#line 18 "examples/slang/sdf_rendering.slang"
[[kernel]] void reference(uint3 tid_0 [[thread_position_in_grid]], TriangleMesh_0 constant* mesh_1 [[buffer(1)]], texture2d<float, access::read_write> targetTexture_1 [[texture(0)]], GlobalParams_0 constant* globalParams_1 [[buffer(0)]])
{

#line 18
    KernelContext_0 kernelContext_0;

#line 18
    (&kernelContext_0)->mesh_0 = mesh_1;

#line 18
    (&kernelContext_0)->targetTexture_0 = targetTexture_1;

#line 18
    (&kernelContext_0)->globalParams_0 = globalParams_1;

    uint2 _S48 = tid_0.xy;

#line 20
    float2 uv_1 = (float2(_S48) + float2(0.5) ) / float2(globalParams_1->targetResolution_0);

#line 20
    RayFrame_0 _S49 = globalParams_1->rayFrame_0;

#line 20
    float2 _S50 = uv_1;

#line 20
    Ray_0 _S51 = RayFrame_rayAt_0(&_S49, &_S50);

#line 20
    float3 device* _S52 = (&kernelContext_0)->mesh_0->vertices_0;

#line 20
    uint3 device* _S53 = (&kernelContext_0)->mesh_0->triangles_0;

#line 20
    Ray_0 _S54 = _S51;

#line 20
    PositionalSphereTrace_0 _S55 = PositionalSphereTrace_sphereTrace_0(_S52, _S53, &_S54, int(100), 0.00100000004749745, 10.0);

#line 36
    if((_S55.t_0) > 0.0)
    {

#line 36
        float3 _S56 = float3(0.5) ;

        (&kernelContext_0)->targetTexture_0.write(float4(_S55.p_0 * _S56 + _S56, 1.0),_S48);

#line 36
    }
    else
    {

#line 43
        (&kernelContext_0)->targetTexture_0.write(float4(0.0, 0.0, 0.0, 1.0),_S48);

#line 36
    }

#line 45
    return;
}

