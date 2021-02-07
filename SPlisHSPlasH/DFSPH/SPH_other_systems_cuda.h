#ifndef SPH_OTHER_SYSTEMS
#define SPH_OTHER_SYSTEMS


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"
#include "SPH_dynamic_window_buffer.h"
#include "RestFLuidLoader.h"
#include "openBoundariesSimple.h"
#include "DynamicWindow.h"

using namespace SPH;

cudaError_t read_last_error_cuda(std::string msg, bool show_message = true);

void get_UnifiedParticleSet_min_max_naive_cuda(SPH::UnifiedParticleSet& particleSet, Vector3d& min, Vector3d& max);

//fluid height related
RealCuda find_fluid_height_cuda(SPH::DFSPHCData& data, SPH::UnifiedParticleSet* particleSet);
void control_fluid_height_cuda(SPH::DFSPHCData& data, RealCuda target_height);



Vector3d get_simulation_center_cuda(SPH::DFSPHCData& data);


//dynamic simulation area related
void move_simulation_cuda(SPH::DFSPHCData& data, Vector3d movement);
void add_border_to_damp_planes_cuda(SPH::DFSPHCData& data, bool x_displacement = true, bool z_displacement = true);


//error checking

//this function will check if there are particle that tunneld throught the boundary box
//3 possible modes:
//	0: just report
//	1: end simulaton if tunneled detected
//	2: rmv particles that have tunneled
int check_particles_positions_cuda(SPH::DFSPHCData& data, int mode=0, bool report=true);


Vector3d get_avg_velocity_cuda(SPH::UnifiedParticleSet* particleSet);


void evaluate_density_field(SPH::DFSPHCData& data, SPH::UnifiedParticleSet* particleSet);

template<class T>
void remove_tagged_particles(SPH::UnifiedParticleSet* particleSet, T* index_array, T* index_array_sorted, int countToRemove,
	bool forceKeepOrder = false, bool runDebug = false);

#endif //DFSPH_STATIC_VAR_STRUCT