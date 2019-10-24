#include "SPH_memory_storage_precomp_kernels.h"

#include "cuda_runtime.h"

__constant__ RealCuda W[SAMPLE_COUNT];
__constant__ RealCuda grad_W[SAMPLE_COUNT];

void writte_to_precomp_kernel(RealCuda* W_i, RealCuda* grad_W_i) {
	cudaMemcpyToSymbol(W, W_i, sizeof(RealCuda) * SAMPLE_COUNT);
	cudaMemcpyToSymbol(grad_W, grad_W_i, sizeof(RealCuda) * SAMPLE_COUNT);
}

__device__ inline RealCuda get_constant_W_cuda(unsigned int pos) {
	return W[pos];
}

__device__ inline RealCuda get_constant_grad_W_cuda(unsigned int pos) {
	return grad_W[pos];
}