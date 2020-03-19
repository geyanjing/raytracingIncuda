#pragma once
#ifndef AABBH
#define AABBH

#include "vec3.h"
#include "ray.h"

class aabb {
public:
	aabb() {}
	__device__ __host__ aabb(const vec3& a, const vec3& b) { _min = a; _max = b; }

	__device__ __host__ vec3 min() const { return _min; }
	__device__ __host__ vec3 max() const { return _max; }

	__device__ bool hit(const ray& r, float tmin, float tmax) const;

	vec3 _min;
	vec3 _max;
};

inline float ffmin(float a, float b) { return a < b ? a : b; }
inline float ffmax(float a, float b) { return a > b ? a : b; }

inline aabb* surrounding_box(aabb box0, aabb box1) {
	vec3 small(ffmin(box0.min().x(), box1.min().x()),
		ffmin(box0.min().y(), box1.min().y()),
		ffmin(box0.min().z(), box1.min().z()));
	vec3 big(ffmax(box0.max().x(), box1.max().x()),
		ffmax(box0.max().y(), box1.max().y()),
		ffmax(box0.max().z(), box1.max().z()));
	return new aabb(small, big);
}

#endif // !AABBH