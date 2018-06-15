#ifndef DFSPH_CUDA
#define DFSPH_CUDA

#include <GL/glew.h>
#include <cuda_gl_interop.h>

class ParticleSetRenderingData {
public:
	cudaGraphicsResource_t pos;
	cudaGraphicsResource_t vel;

	GLuint vaoFluid;
	GLuint vaoBoundaries;
	GLuint pos_buffer;
	GLuint vel_buffer;
};

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


void cuda_initNeighborsSearchDataSet(SPH::UnifiedParticleSet& particleSet, SPH::NeighborsSearchDataSet& dataSet, 
	RealCuda kernel_radius, bool sortBuffers=false);

void cuda_sortData(SPH::UnifiedParticleSet& particleSet, SPH::NeighborsSearchDataSet& neighborsDataSet);




void cuda_renderFluid(SPH::DFSPHCData& data);
void cuda_opengl_initParticleRendering(ParticleSetRenderingData& renderingData, unsigned int numParticles,
	Vector3d** pos, Vector3d** vel);
void cuda_opengl_renderParticleSet(ParticleSetRenderingData& renderingData, unsigned int numParticles);

void cuda_renderBoundaries(SPH::DFSPHCData& data, bool renderWalls);




void allocate_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container);
void load_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container, Vector3d* pos, Vector3d* vel, RealCuda* mass);
void read_rigid_body_force_cuda(SPH::UnifiedParticleSet& container);
void allocate_and_copy_UnifiedParticleSet_vector_cuda(SPH::UnifiedParticleSet** out_vector, SPH::UnifiedParticleSet* in_vector, int numSets);


void allocate_precomputed_kernel_managed(SPH::PrecomputedCubicKernelPerso& kernel, bool minimize_managed = false);
void init_precomputed_kernel_from_values(SPH::PrecomputedCubicKernelPerso& kernel, RealCuda* w, RealCuda* grad_W);


void allocate_neighbors_search_data_set(SPH::NeighborsSearchDataSet& dataSet);
void release_neighbors_search_data_set(SPH::NeighborsSearchDataSet& dataSet, bool keep_result_buffers);


int test_cuda();


#endif