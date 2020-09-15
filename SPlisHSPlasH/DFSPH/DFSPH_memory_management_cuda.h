#ifndef DFSPH_MEMORY_MANAGEMENT_CUDA
#define DFSPH_MEMORY_MANAGEMENT_CUDA

#include "DFSPH_rendering_cuda.h"


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"

void allocate_DFSPHCData_base_cuda(SPH::DFSPHCData& data);
void free_DFSPHCData_base_cuda(SPH::DFSPHCData& data);

void allocate_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container);
void release_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container);

void load_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container, Vector3d* pos, Vector3d* vel, RealCuda* mass);
void read_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& container, Vector3d* pos, Vector3d* vel, RealCuda* mass, Vector3d* pos0 = NULL);
void copy_UnifiedParticleSet_cuda(SPH::UnifiedParticleSet& dst, SPH::UnifiedParticleSet& src, bool copy_warmstart=false);

void read_rigid_body_force_cuda(SPH::UnifiedParticleSet& container);
void compute_fluid_impact_on_dynamic_body_cuda(SPH::UnifiedParticleSet& container, Vector3d& force, Vector3d& moment,
	const Vector3d &reduction_factor);
void compute_fluid_Boyancy_on_dynamic_body_cuda(SPH::UnifiedParticleSet& container, Vector3d& force, Vector3d& pt_appli);
void allocate_and_copy_UnifiedParticleSet_vector_cuda(SPH::UnifiedParticleSet** out_vector, SPH::UnifiedParticleSet* in_vector, int numSets);

void allocate_grouped_neighbors_struct_cuda(SPH::DFSPHCData& data);
void free_grouped_neighbors_struct_cuda(SPH::DFSPHCData& data);

void update_neighborsSearchBuffers_UnifiedParticleSet_vector_cuda(SPH::UnifiedParticleSet** out_vector, SPH::UnifiedParticleSet* in_vector, int numSets);
//this function is the one that must be called when releasing the unified particles that are fully cuda allocated
void release_UnifiedParticleSet_vector_cuda(SPH::UnifiedParticleSet** vector, int numSets);

template<class T> void update_active_particle_number_cuda(T& container);

//this function remove the "nb_to_remove" particles with the highest score
//NOTE: the score must be given in a gpu allocated array
void remove_particles(SPH::UnifiedParticleSet* particleSet, unsigned int* score_gpu_ptr, int nb_to_remove);

void change_fluid_max_particle_number(SPH::DFSPHCData& data, int numParticlesMax);
void change_max_particle_number(SPH::UnifiedParticleSet& container, int numParticlesMax);
void change_max_particle_number(SPH::NeighborsSearchDataSet& dataSet, int numParticlesMax);

void add_particles_cuda(SPH::UnifiedParticleSet& container, int num_additional_particles, const Vector3d* pos, const Vector3d* vel);

template<class T> void set_buffer_to_value(T* buff, T val, int size);
template<class T> void apply_factor_to_buffer(T* buff, T val, int size);
template<class T, int clamping_type> void clamp_buffer_to_value(T* buff, T val, int size);


void allocate_precomputed_kernel_managed(SPH::PrecomputedCubicKernelPerso& kernel, bool minimize_managed = false);
void free_precomputed_kernel_managed(SPH::PrecomputedCubicKernelPerso& kernel);
void init_precomputed_kernel_from_values(SPH::PrecomputedCubicKernelPerso& kernel, RealCuda* w, RealCuda* grad_W);


void allocate_neighbors_search_data_set(SPH::NeighborsSearchDataSet& dataSet, bool result_buffers_only = false,
	bool particle_related_only = false, bool allocate_gpu = true);
void release_neighbors_search_data_set(SPH::NeighborsSearchDataSet& dataSet, bool keep_result_buffers,
	bool keep_grid_related = false, bool keep_gpu = false);


void load_bender2019_boundaries_from_cpu(SPH::UnifiedParticleSet& container, RealCuda* V_rigids_i, Vector3d* X_rigids_i);



#endif //DFSPH_MEMORY_MANAGEMENT_CUDA