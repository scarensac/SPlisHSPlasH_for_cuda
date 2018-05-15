#ifndef DFSPH_CUDA
#define DFSPH_CUDA


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"

void cuda_divergence_warmstart_init(SPH::DFSPHCData& data);
template<bool warmstart> void cuda_divergence_compute(SPH::DFSPHCData& data);
void cuda_divergence_init(SPH::DFSPHCData& data);//also compute densities and factors
RealCuda cuda_divergence_loop_end(SPH::DFSPHCData& data);//reinit the densityadv and calc the error

void cuda_viscosityXSPH(SPH::DFSPHCData& data);

void cuda_CFL(SPH::DFSPHCData& data, const RealCuda minTimeStepSize, RealCuda m_cflFactor, RealCuda m_cflMaxTimeStepSize);

void cuda_update_vel(SPH::DFSPHCData& data);

template<bool warmstart> void cuda_pressure_compute(SPH::DFSPHCData& data); 
void cuda_pressure_init(SPH::DFSPHCData& data);
RealCuda cuda_pressure_loop_end(SPH::DFSPHCData& data);

void cuda_update_pos(SPH::DFSPHCData& data);

//Return the number of iterations
int cuda_divergenceSolve(SPH::DFSPHCData& data, const unsigned int maxIter, const RealCuda maxError);
int cuda_pressureSolve(SPH::DFSPHCData& data, const unsigned int maxIter, const RealCuda maxError);

//those functions are for the neighbors search
void cuda_neighborsSearch(SPH::DFSPHCData& data);

void cuda_initNeighborsSearchDataSet(SPH::DFSPHCData& data, SPH::NeighborsSearchDataSet& dataSet, bool is_boundaries);
void cuda_sortData(SPH::DFSPHCData& data, SPH::NeighborsSearchDataSet& neighborsDataSet, bool is_boundaries=false);


void cuda_renderFluid(SPH::DFSPHCData& data);
void cuda_opengl_initFluidRendering(SPH::DFSPHCData& data);
void cuda_opengl_renderFluid(SPH::DFSPHCData& data);

void cuda_renderBoundaries(SPH::DFSPHCData& data);
void cuda_opengl_renderBoundaries(SPH::DFSPHCData& data);








void allocate_c_array_struct_cuda_managed(SPH::DFSPHCData& data, bool minimize_managed = false);
void reset_c_array_struct_cuda_from_values(SPH::DFSPHCData& data, Vector3d* posBoundary, Vector3d* velBoundary,
	RealCuda* boundaryPsi, Vector3d* posFluid, Vector3d* velFluid, RealCuda* mass);

void allocate_precomputed_kernel_managed(SPH::PrecomputedCubicKernelPerso& kernel, bool minimize_managed = false);
void init_precomputed_kernel_from_values(SPH::PrecomputedCubicKernelPerso& kernel, RealCuda* w, RealCuda* grad_W);


void allocate_neighbors_search_data_set(SPH::NeighborsSearchDataSet& dataSet);
void release_neighbors_search_data_set(SPH::NeighborsSearchDataSet& dataSet, bool keep_result_buffers);


int test_cuda();


#endif