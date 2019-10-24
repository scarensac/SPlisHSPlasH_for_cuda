#ifndef SPH_memory_storage_precomp_kernels_cuh
#define SPH_memory_storage_precomp_kernels_cuh


#include "SPlisHSPlasH\BasicTypes.h"
#include <string>
#include <vector>



#include "SPlisHSPlasH\Vector.h"
#include "SPlisHSPlasH\Quaternion.h"

#include "DFSPH_define_c.h"
#include "cuda_runtime.h"

__device__ RealCuda get_constant_W_cuda(const SPH::Vector3d &r);// { return 0; }
__device__ inline RealCuda get_constant_W_cuda(const RealCuda r) { return 0; }
__device__ inline SPH::Vector3d get_constant_grad_W_cuda(const SPH::Vector3d &r) { return SPH::Vector3d(0); }


#endif