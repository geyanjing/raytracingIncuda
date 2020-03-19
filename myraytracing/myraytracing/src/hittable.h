#ifndef HITTABLEH
#define HITTABLEH
//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#include "ray.h"
#include "aabb.h"
#include "device_launch_parameters.h"

class material;

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	material* mat_ptr;
};

//class hittable  {
//    public:
//		__device__  virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
//		 virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
//		//�����ƶ�������Ҫtime1��time2����֡�ļ���ͱ߽�򽫽��ͨ���ü���ƶ��Ķ���
//		 //__device__ virtual bool isleaf()const;//���
//		 bool leaf = true;
//};

#endif
