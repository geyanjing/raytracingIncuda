#include "aabb.h"
#include "ray.h"

__device__ bool aabb::hit(const ray& r, float tmin, float tmax) const {
	for (int a = 0; a < 3; a++) {
		float invD = 1.0f / r.direction()[a];
		float t0 = (min()[a] - r.origin()[a]) * invD;
		float t1 = (max()[a] - r.origin()[a]) * invD;
		if (invD < 0.0f) {
			float t = t0; t0 = t1; t1 = t;
		}
		//std::swap(t0, t1);
		tmin = t0 > tmin ? t0 : tmin;
		tmax = t1 < tmax ? t1 : tmax;
		if (tmax <= tmin) {
			return false;
		}
	}
	return true;
}