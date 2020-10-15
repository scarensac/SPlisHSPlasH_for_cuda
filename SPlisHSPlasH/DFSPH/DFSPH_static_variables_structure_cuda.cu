#include "DFSPH_static_variables_structure_cuda.h"



#include <curand.h>
#include <curand_kernel.h>


//Static Variable Structure kernels
__global__ void initCurand_kernel(curandState *state) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 1) { return; }

	curand_init(1234, 0, 0, state);
}

SVS_CU* SVS_CU::get(bool free) {
	static SVS_CU* v = NULL;

	if (free) {
		if (v != NULL) {
			delete v; v = NULL;
		}
	}
	else {
		if (v == NULL) {
			v = new SVS_CU();
		}
	}

	return v;
}

SVS_CU::SVS_CU() {
	cudaMallocManaged(&(avg_density_err), sizeof(RealCuda));
	shuffle_index = NULL;


	cudaMalloc(&(curand_state), sizeof(curandState));
	initCurand_kernel << <1, 1 >> > (curand_state);
	gpuErrchk(cudaDeviceSynchronize());

	cudaMallocManaged(&(count_rmv_particles), sizeof(int));
	cudaMallocManaged(&(count_possible_particles), sizeof(int));
	cudaMallocManaged(&(count_moved_particles), sizeof(int));
	cudaMallocManaged(&(count_invalid_position), sizeof(int));

	cudaMallocManaged(&(column_max_height), CELL_ROW_LENGTH*CELL_ROW_LENGTH * sizeof(RealCuda));
	cudaMallocManaged(&(tagged_particles_count), sizeof(int));
	cudaMallocManaged(&(count_created_particles), sizeof(int));

	cudaMallocManaged(&(force_cuda), sizeof(Vector3d));
	cudaMallocManaged(&(moment_cuda), sizeof(Vector3d));
	cudaMallocManaged(&(pt_cuda), sizeof(Vector3d));
}

SVS_CU::~SVS_CU() {
	CUDA_FREE_PTR(avg_density_err);
	CUDA_FREE_PTR(shuffle_index);
	CUDA_FREE_PTR(curand_state);
	CUDA_FREE_PTR(count_rmv_particles);
	CUDA_FREE_PTR(count_possible_particles);
	CUDA_FREE_PTR(count_moved_particles);
	CUDA_FREE_PTR(count_invalid_position);
	CUDA_FREE_PTR(column_max_height);
	CUDA_FREE_PTR(tagged_particles_count);
	CUDA_FREE_PTR(count_created_particles);
	CUDA_FREE_PTR(force_cuda);
	CUDA_FREE_PTR(moment_cuda);
	CUDA_FREE_PTR(pt_cuda);
}

void SVS_CU::particleNumberChanged() {
	CUDA_FREE_PTR(get()->shuffle_index);
}