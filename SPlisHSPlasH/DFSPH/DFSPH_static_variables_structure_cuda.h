#ifndef DFSPH_STATIC_VAR_STRUCT
#define DFSPH_STATIC_VAR_STRUCT

//!!!!!! warning this file should no need to be include in another .h file !!!!!!

#include "DFSPH_define_cuda.h"
#include "DFSPH_macro_cuda.h"

#include "SPlisHSPlasH\Vector.h"

using namespace SPH;


struct curandStateXORWOW;
using curandState = struct curandStateXORWOW;

//Static Variable Structure
class SVS_CU {
public:
	static SVS_CU* get(bool free = false);

	SVS_CU();
	virtual ~SVS_CU();

	static void particleNumberChanged();

	//cuda variables
	RealCuda* avg_density_err;
	unsigned int* shuffle_index;
	curandState *curand_state;
	int* count_rmv_particles;
	int* count_possible_particles;
	int* count_moved_particles;
	int* count_invalid_position;
	RealCuda* column_max_height;
	int* tagged_particles_count;
	int* count_created_particles;
	Vector3d* force_cuda;
	Vector3d* moment_cuda;
	Vector3d* pt_cuda;


	//cpu variables

};

#endif //DFSPH_STATIC_VAR_STRUCT