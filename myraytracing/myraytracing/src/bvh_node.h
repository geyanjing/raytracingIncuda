#pragma once
#ifndef BVHNODEH
#define BVHNODEH

#include "sphere.h"
#include "random.h"
#include <vector>
#include<list>

class bvh_node {
public:
	__device__ __host__ bvh_node() {}
	//__device__ bvh_node(hittable* l, hittable* r, aabb x) { left = l; right = r; box = x; }
	//bvh_node(sphere** l, int n, float time0, float time1);
	/*__device__  bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const ;*/
	//__device__ virtual bool isleaf()const;
   // bool bounding_box(float t0, float t1, aabb* box);

	bvh_node* left;
	bvh_node* right;
	aabb* box;
	sphere* sph;
	bool leaf;
};

//bvh_node* ToDevice(bvh_node*);
//请注意，子代指针指向通用的命中表。
//它们可以是other bvh_nodes或 spheresor或任何其他hittable。
bvh_node* buildleaf(sphere* sph) {
	auto bvh = new bvh_node();
	bvh->leaf = true;
	bvh->sph = sph;
	bvh->left = nullptr;
	bvh->right = nullptr;
	//aabb* thebox;
	sph->bounding_box(0, 0, &(bvh->box));
	//printf("leftr:  %p  %i\n", bvh->box,i++);

	//printf("%f", bvh->box->min().x());
	//bvh->box = thebox;
	return bvh;
}

//bool bvh_node::bounding_box(float t0, float t1, aabb* b){
//	b = box;
//	return true;
//}

//__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
//	if (box.hit(r, t_min, t_max)) {
//		hit_record left_rec, right_rec;
//		bool hit_left = left->hit(r, t_min, t_max, left_rec);
//		bool hit_right = right->hit(r, t_min, t_max, right_rec);
//		if (hit_left && hit_right) {
//			if (left_rec.t < right_rec.t)
//				rec = left_rec;
//			else
//				rec = right_rec;
//			return true;
//		}
//		else if (hit_left) {
//			rec = left_rec;
//			return true;
//		}
//		else if (hit_right) {
//			rec = right_rec;
//			return true;
//		}
//		else
//			return false;
//	}
//	else return false;
//}
//__device__ bool bvh_node::isleaf()const {
//	if (leaf)
//		return true;
//	else
//		return false;
//}

int box_x_compare(const void* a, const void* b) {
	aabb* box_left = nullptr, * box_right = nullptr;
	sphere* ah = *(sphere**)a;
	sphere* bh = *(sphere**)b;

	if (!ah->bounding_box(0, 0, &box_left) || !bh->bounding_box(0, 0, &box_right))
		std::cerr << "no bounding box in bvh_node constructor\n";
	if ((box_left != nullptr) && (box_right != nullptr)) {
		if ((box_left->min()).x() - (box_right->min()).x() < 0.0) {
			/*free(box_left);
			free(box_right);
			free(ah);
			free(bh);*/
			return -1;
		}

		else {
			/*free(box_left);
			free(box_right);
			free(ah);
			free(bh);*/
			return 1;
		}
	}
}

int box_y_compare(const void* a, const void* b) {
	aabb* box_left = nullptr, * box_right = nullptr;
	sphere* ah = *(sphere**)a;
	sphere* bh = *(sphere**)b;

	if (!ah->bounding_box(0, 0, &box_left) || !bh->bounding_box(0, 0, &box_right))
		std::cerr << "no bounding box in bvh_node constructor\n";

	if ((box_left != nullptr) && (box_right != nullptr)) {
		if ((box_left->min()).y() - (box_right->min()).y() < 0.0) {
			/*free(box_left);
			free(box_right);
			free(ah);
			free(bh);*/
			return -1;
		}

		else {
			/*free(box_left);
			free(box_right);
			free(ah);
			free(bh);*/
			return 1;
		}
	}
}

int box_z_compare(const void* a, const void* b) {
	aabb* box_left = nullptr, * box_right = nullptr;
	sphere* ah = *(sphere**)a;
	sphere* bh = *(sphere**)b;

	if (!ah->bounding_box(0, 0, &box_left) || !bh->bounding_box(0, 0, &box_right))
		std::cerr << "no bounding box in bvh_node constructor\n";

	if ((box_left != nullptr) && (box_right != nullptr)) {
		if ((box_left->min()).z() - (box_right->min()).z() < 0.0) {
			return -1;
		}

		else {
			return 1;
		}
	}
}

bvh_node* BuildBVH(sphere** list, int n, float time0, float time1)
{
	auto bvh = new bvh_node();
	bvh->leaf = false;

	if (n == 1)return buildleaf(list[0]);
	if (n == 2)
	{
		//printf("二分\n");
		bvh->sph = nullptr;
		bvh->left = buildleaf(list[0]);

		bvh->right = buildleaf(list[1]);
	}
	else {
		const auto axis = int(3 * random_double());
		if (axis == 0) qsort(list, n, sizeof(sphere*), box_x_compare);
		else if (axis == 1) qsort(list, n, sizeof(sphere*), box_y_compare);
		else  qsort(list, n, sizeof(sphere*), box_z_compare);

		bvh->sph = nullptr;
		bvh->left = BuildBVH(list, n / 2, 0, 0);
		bvh->right = BuildBVH(list + n / 2, n - n / 2, 0, 0);
	}
	if (bvh->left->box && bvh->right->box) {
		bvh->box = surrounding_box(*(bvh->left->box), *(bvh->right->box));
		//printf("leftr:  %p  %i\n", bvh->box, i++);
	}

	return bvh;
}

#endif // !BVHNDEH