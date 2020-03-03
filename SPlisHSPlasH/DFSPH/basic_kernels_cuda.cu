#include "basic_kernels_cuda.cuh"

#include <curand.h>
#include <curand_kernel.h>

__global__ void DFSPH_setVector3dBufferToZero_kernel(Vector3d* buff, unsigned int buff_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= buff_size) { return; }

	buff[i] = Vector3d(0, 0, 0);
}

template<class T> __global__ void cuda_setBufferToValue_kernel(T* buff, T value, unsigned int buff_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= buff_size) { return; }

	buff[i] = value;
}
template __global__ void cuda_setBufferToValue_kernel<Vector3d>(Vector3d* buff, Vector3d value, unsigned int buff_size);
template __global__ void cuda_setBufferToValue_kernel<int>(int* buff, int value, unsigned int buff_size);
template __global__ void cuda_setBufferToValue_kernel<RealCuda>(RealCuda* buff, RealCuda value, unsigned int buff_size);


__global__ void DFSPH_Histogram_kernel(unsigned int* in, unsigned int* out, unsigned int num_particles) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_particles) { return; }

	atomicAdd(&(out[in[i]]), 1);

}

__global__ void DFSPH_setBufferValueToItself_kernel(unsigned int* buff, unsigned int buff_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= buff_size) { return; }

	buff[i] = i;
}

__global__ void apply_delta_to_buffer_kernel(Vector3d* buffer, Vector3d delta, const unsigned int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size) { return; }

	buffer[i] += delta;
}

template<class T>
__global__ void fillRandom_kernel(unsigned int *buff, unsigned int nbElements, T min, T max, curandState *state) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 1) { return; }

	curandState localState = *state;
	for (int j = 0; j < nbElements; ++j) {
		T x = curand(&localState);
		x *= (max - min);
		x += min;
		buff[i] = x;
	}
	*state = localState;
}

