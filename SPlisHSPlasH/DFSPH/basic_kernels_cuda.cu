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


template<class T> __global__ void cuda_applyFactorToBuffer_kernel(T* buff, T value, unsigned int buff_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= buff_size) { return; }

	buff[i] *= value;
}
template __global__ void cuda_applyFactorToBuffer_kernel<Vector3d>(Vector3d* buff, Vector3d value, unsigned int buff_size);
template __global__ void cuda_applyFactorToBuffer_kernel<int>(int* buff, int value, unsigned int buff_size);
template __global__ void cuda_applyFactorToBuffer_kernel<RealCuda>(RealCuda* buff, RealCuda value, unsigned int buff_size);


//note for the clamping type
//the main problem is that there are multiples ways to clamp a number and depending on the type of data there may be more
//for now I need the vector 3D so there are the 4 obvious min, max, absolute value(min and max) ; but also a clamping on the length of the vector
//also making it generic is probably impossible since the setter are different for the vector3D
//so I need a specialized kernel for the vector3D
//so for now here is how that parameter works
// 0 : keep anything below the parameter
// 1 : keep anything above the parameter
// 2 : keep anything below the absolute value
// 3 : keep anything above the absolute value
// 4 : vector special: if norm above value normalize it to the value, Ill read the first cell of the vector to know the clamping value
/// TODO impelemnt it all, for now I only need the 2 and 4 so I'll tag the others with an asm("trap")
template<int clamping_type> __global__ void cuda_clampV3dBufferToValue_kernel(Vector3d* buff, Vector3d value, unsigned int buff_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= buff_size) { return; }

	if (clamping_type==2) {
		Vector3d v = buff[i];
		v.toMin(value);
		value *= -1;
		v.toMax(value);
		buff[i] = v;
	}else if (clamping_type == 4) {
		RealCuda l = buff[i].norm();
		if (l > value.x) {
			buff[i] *= value.x / l;
		}
	}else {
		asm("trap;");
	}

	buff[i] = value;
}
template __global__ void cuda_clampV3dBufferToValue_kernel<0>(Vector3d* buff, Vector3d value, unsigned int buff_size);
template __global__ void cuda_clampV3dBufferToValue_kernel<1>(Vector3d* buff, Vector3d value, unsigned int buff_size);
template __global__ void cuda_clampV3dBufferToValue_kernel<2>(Vector3d* buff, Vector3d value, unsigned int buff_size);
template __global__ void cuda_clampV3dBufferToValue_kernel<3>(Vector3d* buff, Vector3d value, unsigned int buff_size);
template __global__ void cuda_clampV3dBufferToValue_kernel<4>(Vector3d* buff, Vector3d value, unsigned int buff_size);



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

