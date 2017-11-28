#ifndef DFSPH_CUDA
#define DFSPH_CUDA


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"


int test_cuda();

void allocate_c_array_struct_cuda_managed(SPH::DFSPHCData& data);

#endif