#ifndef DFSPH_CORE_CUDA
#define DFSPH_CORE_CUDA



#include "DFSPH_rendering_cuda.h"
#include "DFSPH_memory_management_cuda.h"


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"

using namespace SPH;


////////////////////////////////////////////////////
/////////       DIVERGENCE SOLVER      /////////////
////////////////////////////////////////////////////

//internal functions
void cuda_divergence_warmstart_init(SPH::DFSPHCData& data);
template<bool warmstart> void cuda_divergence_compute(SPH::DFSPHCData& data);
void cuda_divergence_init(SPH::DFSPHCData& data);//also compute densities and factors
RealCuda cuda_divergence_loop_end(SPH::DFSPHCData& data);//reinit the densityadv and calc the error

//actual function, return the nbr of iteration 
int cuda_divergenceSolve(SPH::DFSPHCData& data, const unsigned int maxIter, const RealCuda maxError);


////////////////////////////////////////////////////
/////////          DENSITY SOLVER      /////////////
////////////////////////////////////////////////////

//internal functions
template<bool warmstart> void cuda_pressure_compute(SPH::DFSPHCData& data);
void cuda_pressure_init(SPH::DFSPHCData& data);
RealCuda cuda_pressure_loop_end(SPH::DFSPHCData& data);

//actual function, return the nbr of iteration 
int cuda_pressureSolve(SPH::DFSPHCData& data, const unsigned int maxIter, const RealCuda maxError);


////////////////////////////////////////////////////
/////////         EXTERNAL FORCES      /////////////
////////////////////////////////////////////////////


void cuda_externalForces(SPH::DFSPHCData& data);


////////////////////////////////////////////////////
/////////         NEIGHBORS SEARCH     /////////////
////////////////////////////////////////////////////

void cuda_initNeighborsSearchDataSet(SPH::UnifiedParticleSet& particleSet, SPH::NeighborsSearchDataSet& dataSet,
	DFSPHCData &data, bool sortBuffers = false);
void cuda_initNeighborsSearchDataSetGroupedDynamicBodies(SPH::DFSPHCData& data);
void cuda_sortData(SPH::UnifiedParticleSet& particleSet, unsigned int * sort_id);
void cuda_shuffleData(SPH::UnifiedParticleSet& particleSet);

void cuda_neighborsSearch(SPH::DFSPHCData& data, bool need_sort);

///WARNING:: this function requires having run the init neighbors search data before execution
void cuda_updateNeighborsStorage(SPH::DFSPHCData& data, SPH::UnifiedParticleSet& particleSet, int iter_debug=-1);


////////////////////////////////////////////////////
/////////             OTHERS           /////////////
////////////////////////////////////////////////////


void cuda_update_vel(SPH::DFSPHCData& data);
void cuda_update_pos(SPH::DFSPHCData& data);
void cuda_CFL(SPH::DFSPHCData& data, const RealCuda minTimeStepSize, RealCuda m_cflFactor, RealCuda m_cflMaxTimeStepSize);


//this function is used to update the location of the dynamic bodies articles
//this function allow to move the simulation location by moving the boundaires particles and adapting the fluid
//the particles associated with rigid objects are untouched
void update_dynamicObject_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& particle_set);


void compute_UnifiedParticleSet_particles_mass_cuda(SPH::DFSPHCData& data, SPH::UnifiedParticleSet& container);




#endif //DFSPH_CORE_CUDA

