#ifndef DFSPH_CUDA
#define DFSPH_CUDA


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"

//compute the density of the particles 
void cuda_compute_density(SPH::DFSPHCData& data);

void cuda_computeDFSPHFactor(SPH::DFSPHCData& data);

//use the algo XSPH to simulate the viscosity 
//start by setting the acc to the gravitation this remove the need of an addictional kernel to do it
void cuda_viscosity_XSPH(SPH::DFSPHCData& data);

void cuda_updateVelocities(SPH::DFSPHCData& data);
void cuda_updatePositions(SPH::DFSPHCData& data);

//the next functions are for the internal working of the dfsph algortihm
//the first function execute the whole algorithm and the others are for partial integration
//in the cpu algorithm
//void cuda_divergenceSolve(SPH::DFSPHCData& data);
void cuda_divergenceSolve_warmStart_firstLoop(SPH::DFSPHCData& data);
void cuda_divergenceSolve_warmStart_secondLoop(SPH::DFSPHCData& data);
void cuda_divergenceSolve_initialize(SPH::DFSPHCData& data);











void allocate_c_array_struct_cuda_managed(SPH::DFSPHCData& data);
void allocate_precomputed_kernel_managed(SPH::PrecomputedCubicKernelPerso& kernel);



int test_cuda();


#endif