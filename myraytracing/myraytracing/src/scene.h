#ifndef SCENEH
#define SCENEH

#include "bvh_node.h"
#include "material.h"

bvh_node* random_scene() {
	int n = 500;
	sphere** list = new sphere * [n + 1];
	float d[6] = { 0.5, 0.5, 0.5 };
	list[0] = new sphere(vec3(0, -1000, 0), 1000, new material(lambertian, d));
	int i = 1;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = random_double();
			vec3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {
					float d[6] = { random_double() * random_double(),
							random_double() * random_double(),
							random_double() * random_double() };
					list[i++] = new sphere(center, 0.2, new material(lambertian, d));
				}
				else if (choose_mat < 0.95) { // metal
					float d[6] = { 0.5 * (1 + random_double()),
							0.5 * (1 + random_double()),
							0.5 * (1 + random_double()),
							0.5 * random_double() };
					list[i++] = new sphere(
						center, 0.2,
						new material(metal, d));
				}
				else {  // glass
					float d[6] = { 1.5 };
					list[i++] = new sphere(center, 0.2, new material(dielectirc, d));
				}
			}
		}
	}
	float a[6] = { 1.5 };
	float b[6] = { 0.4,0.2,0.1 };
	float c[6] = { 0.7,0.6,0.6,0.0 };
	list[i++] = new sphere(vec3(0, 1, 0), 1.0, new material(dielectirc, a));
	list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new material(lambertian, b));
	list[i++] = new sphere(vec3(4, 1, 0), 1.0, new material(metal, c));

	//return new hittable_list(list, i);

	return BuildBVH(list, i, 0.0, 0.0);
}
#endif // !SCENEH

