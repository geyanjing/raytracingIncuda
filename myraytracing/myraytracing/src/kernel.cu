#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <time.h>
#include "scene.h"
#include "camera.h"

#include "random.h"

#include "myhit.h"
#include <float.h>

//limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

bvh_node* ToDevice(bvh_node* host)
{
	if (host == nullptr)return nullptr;

	bvh_node* device;
	bvh_node* bvh = new bvh_node();

	bvh->leaf = host->leaf;
	if (host->sph == nullptr)bvh->sph = nullptr;
	else
	{
		sphere* sphere1 = new sphere();
		sphere1->center = host->sph->center;
		sphere1->radius = host->sph->radius;
		checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&sphere1->mat_ptr), sizeof(material)));
		checkCudaErrors(cudaMemcpy(sphere1->mat_ptr, host->sph->mat_ptr, sizeof(material), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&bvh->sph), sizeof(sphere)));
		checkCudaErrors(cudaMemcpy(bvh->sph, sphere1, sizeof(sphere), cudaMemcpyHostToDevice));
	}

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&bvh->box), sizeof(aabb)));
	checkCudaErrors(cudaMemcpy(bvh->box, host->box, sizeof(aabb), cudaMemcpyHostToDevice));

	bvh->left = ToDevice(host->left);
	bvh->right = ToDevice(host->right);

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(bvh_node)));
	checkCudaErrors(cudaMemcpy(device, bvh, sizeof(bvh_node), cudaMemcpyHostToDevice));
	return device;
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ bool myhit(const ray& r, float t_min, float t_max, hit_record& rec, bvh_node* bvh) {
	bvh_node* stack[20];
	int ptr = 0;
	bvh_node* current = bvh;
	//int i = 0;
	float mint = t_max;
	//hit_record hit;
	hit_record besthit;

	do
	{
		bvh_node* left = current->left;
		bvh_node* right = current->right;
		if (current->leaf)
		{
			//printf("true\n");

			if (current->sph->hit(r, t_min, mint, besthit)) {
				//besthit = hit;
				mint = besthit.t;
			}
			else {
				if (ptr <= 0)
					current = nullptr;
				else
					current = stack[--ptr];
			}
			//printf("n\n");
		}
		else
		{
			//printf("false\n");
			if (current->box->hit(r, t_min, t_max))
			{
				current = left;
				stack[ptr++] = right;
			}
			else
			{
				//printf("false");
				if (ptr <= 0)current = nullptr;
				else  current = stack[--ptr];
			}
			//printf("%d\n", ++i);
		}
	} while (current != nullptr);

	/*for (int i = 0; i < 20; i++)
		free(stack[i]);
	free(current);*/
	//加上上方为什么不对
	if (mint < t_max) {
		rec = besthit;
		return true;
	}

	else {
		return false;
	}
}
__device__ vec3 color(const ray& r, bvh_node* world, curandState* local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;

		//if ((world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {//hit是递归的，需要转换
		if (myhit(cur_ray, 0.001f, FLT_MAX, rec, world)) {
			printf("s");
			ray scattered;
			vec3 attenuation;

			//ambertian mar = lambertian(vec3(0.9, 0.2, 0.1));

			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				//if ((rec.mat_ptr)->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else {
			printf("v");
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera* cam, bvh_node* world, curandState* rand_state) {
	//printf("rrrrrr");
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col /= float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

//void Print(bvh_node* bvh, bool root)
//{
//	if (root)
//	{
//		printf("BVH Root (%f,%f,%f) ~ (%f,%f,%f)\n", bvh->box->min().x(), bvh->box->min().y(), bvh->box->min().z(), bvh->box->max().x(), bvh->box->max().y(), bvh->box->max().z());
//		return;
//	}
//	if (bvh->sph == nullptr)
//	{
//		printf("Node [isTriangle: %s , AABB: (%f,%f,%f)(%f,%f,%f), sphere(%p)\n", bvh->leaf ? "true" : "false",
//			bvh->box->min().x(), bvh->box->min().y(), bvh->box->min().z(), bvh->box->max().x(), bvh->box->max().y(), bvh->box->max().z(), bvh->sph);
//	}
//	else
//	{
//		printf("Leaf [isTriangle: %s , AABB: (%f,%f,%f)(%f,%f,%f)\n", bvh->leaf ? "true" : "false",
//			bvh->box->min().x(), bvh->box->min().y(), bvh->box->min().z(), bvh->box->max().x(), bvh->box->max().y(), bvh->box->max().z()
//			//bvh->triangle->v1.point.x, bvh->triangle->v1.point.y, bvh->triangle->v1.point.z,
//			//bvh->triangle->v2.point.x, bvh->triangle->v2.point.y, bvh->triangle->v2.point.z,
//			//bvh->triangle->v3.point.x, bvh->triangle->v3.point.y, bvh->triangle->v3.point.z
//		);
//	}
//}

//bvh_node* random_scene() {
//	int n = 500;
//	sphere** list = new sphere * [n + 1];
//	list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
//	int i = 1;
//	for (int a = -11; a < 11; a++) {
//		for (int b = -11; b < 11; b++) {
//			float choose_mat = random_double();
//			vec3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
//			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
//				if (choose_mat < 0.8) {  // diffuse
//					/*list[i++] = new moving_sphere(
//						center, center + vec3(0, 0.5 * random_double(), 0),
//						0.0, 1.0, 0.2,
//						new lambertian(vec3(random_double() * random_double(),
//							random_double() * random_double(),
//							random_double() * random_double()))*/
//					list[i++] = new sphere(center, 0.2,
//						new lambertian(vec3(random_double() * random_double(),
//							random_double() * random_double(),
//							random_double() * random_double())
//						)
//					);
//				}
//				else if (choose_mat < 0.95) { // metal
//					list[i++] = new sphere(
//						center, 0.2,
//						new metal(vec3(0.5 * (1 + random_double()),
//							0.5 * (1 + random_double()),
//							0.5 * (1 + random_double())),
//							0.5 * random_double())
//					);
//				}
//				else {  // glass
//					list[i++] = new sphere(center, 0.2, new dielectric(1.5));
//				}
//			}
//		}
//	}
//
//	list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
//	list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
//	list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
//
//	//return new hittable_list(list, i);
//	printf("%i\n", i);
//	return BuildBVH(list, i, 0.0, 0.0);
//}

int main() {
	int nx = 1200;
	int ny = 800;
	int ns = 10;
	int tx = 8;
	int ty = 8;
	cudaError_t cudaStatus;
	std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";
	//printf("mmmmm");

	vec3 lookfrom(13, 2, 3);
	vec3 lookat(0, 0, 0);
	float dist_to_focus = 10.0;
	float aperture = 0.1;

	camera* cam = new camera(lookfrom, lookat, vec3(0, 1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);
	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);
	bvh_node* world = random_scene();
	//Print(world, true);
	// allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	// allocate random state
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
	/*curandState* d_rand_state2;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));*/

	// we need that 2nd random state to be initialized for the world creation
	/*rand_init << <1, 1 >> > (d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());*/

	// make our world of hitables & the camera
	/*hittable** d_list;
	int num_hitables = 22 * 22 + 1 + 3;
	checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hittable*)));*/
	bvh_node* d_world;
	//checkCudaErrors(cudaMallocManaged((void**)&d_world, sizeof(bvh_node) *1000));
	d_world = ToDevice(world);

	camera* d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera)));
	//create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
	/*checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());*/
	//checkCudaErrors(cudaMemcpy(d_world, world,1000* sizeof(bvh_node), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_camera, cam, sizeof(camera), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_camera, cam, sizeof(camera), cudaMemcpyHostToDevice));

	clock_t start, stop;
	start = clock();
	// Render our buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);

	checkCudaErrors(cudaGetLastError());
	//cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";
	//checkCudaErrors(cudaMemcpy(world, d_world, 1000 * sizeof(bvh_node), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(cam, d_camera, sizeof(camera), cudaMemcpyDeviceToHost));
	// Output FB as Image
	std::ofstream fout(".\\result\\bvh.ppm");
	fout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j * nx + i;

			int ir = int(255.99 * fb[pixel_index].r());
			int ig = int(255.99 * fb[pixel_index].g());
			int ib = int(255.99 * fb[pixel_index].b());
			fout << ir << " " << ig << " " << ib << "\n";
		}
	}

	// clean up
	checkCudaErrors(cudaDeviceSynchronize());
	//free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	//checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
}
