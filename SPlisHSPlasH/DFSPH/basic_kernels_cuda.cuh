#ifndef BASIC_KERNELS
#define BASIC_KERNELS

#include "SPlisHSPlasH\Vector.h"
#include "cuda_runtime.h"

using namespace SPH;

__global__ void DFSPH_setVector3dBufferToZero_kernel(Vector3d* buff, unsigned int buff_size);

template<class T> 
__global__ void cuda_setBufferToValue_kernel(T* buff, T value, unsigned int buff_size);

template<class T>
__global__ void cuda_applyFactorToBuffer_kernel(T* buff, T value, unsigned int buff_size);

template<int clamping_type> 
__global__ void cuda_clampV3dBufferToValue_kernel(Vector3d* buff, Vector3d value, unsigned int buff_size);

template<class T, class T2>
__global__ void cuda_copyBufferCrossType_kernel(T* out, T2* in, unsigned int size);

__global__ void DFSPH_Histogram_kernel(unsigned int* in, unsigned int* out, unsigned int num_particles);

__global__ void DFSPH_setBufferValueToItself_kernel(unsigned int* buff, unsigned int buff_size);

__global__ void apply_delta_to_buffer_kernel(Vector3d* buffer, Vector3d delta, const unsigned int size);

struct curandStateXORWOW;
using curandState = struct curandStateXORWOW;

template<class T>
__global__ void fillRandom_kernel(unsigned int *buff, unsigned int nbElements, T min, T max, curandState *state);




#endif //BASIC_KERNELSs