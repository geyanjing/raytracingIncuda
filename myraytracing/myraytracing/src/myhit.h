#pragma once
#ifndef MYHIT
#define MYHIT

#include "vec3.h"

class material;

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	material* mat_ptr;
};

#endif // !MYHIT
