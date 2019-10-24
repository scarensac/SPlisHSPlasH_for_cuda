#ifndef SPH_memory_storage_precomp_kernels_h
#define SPH_memory_storage_precomp_kernels_h

#include "SPlisHSPlasH\BasicTypes.h"
#include <string>
#include <vector>



#include "SPlisHSPlasH\Vector.h"
#include "SPlisHSPlasH\Quaternion.h"

#include "DFSPH_define_c.h"


void writte_to_precomp_kernel(RealCuda* W_i, RealCuda* gradW_i, RealCuda radius, RealCuda radius2, RealCuda invStepSize);
void test_constant_mem_precomp_kernel_cuda();

#endif