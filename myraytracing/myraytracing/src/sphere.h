#ifndef SPHEREH
#define SPHEREH
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

//#include "hittable.h"
#include "aabb.h"
#include "vec3.h"
//#include "material.h"
#include "myhit.h"

class sphere {
public:
	sphere() {}
	sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m) {};
	__device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

	bool bounding_box(float t0, float t1, aabb** box) {
		*box = new aabb(center - vec3(radius, radius, radius),
			center + vec3(radius, radius, radius));

		return true;
	}

	vec3 center;
	float radius;
	material* mat_ptr;
	//bool leaf = true;
};

//运动模糊的球
//class moving_sphere  {
//public:
//	moving_sphere() {}
//	moving_sphere(vec3 cen0, vec3 cen1, float t0, float t1, float r, material* m)
//		: center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m)
//	{};
//	__device__  bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
//	//virtual bool bounding_box(float t0, float t1, aabb& box) const;
//	__device__ __host__ vec3 center(float time) const;
//	bool bounding_box(float t0, float t1, aabb& box) const {
//		aabb box0(center(t0) - vec3(radius, radius, radius),
//			center(t0) + vec3(radius, radius, radius));
//		aabb box1(center(t1) - vec3(radius, radius, radius),
//			center(t1) + vec3(radius, radius, radius));
//		box = surrounding_box(box0, box1);
//		return true;
//	}
//
//	vec3 center0, center1;
//	float time0, time1;
//	float radius;
//	material* mat_ptr;
//	//bool leaf = true;
//};
//
//__device__ __host__  inline vec3 moving_sphere::center(float time) const {
//	return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
//}

#endif
