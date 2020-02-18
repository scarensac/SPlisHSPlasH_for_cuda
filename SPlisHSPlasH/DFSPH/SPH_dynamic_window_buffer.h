#ifndef SPH_DYNAMIC_WINDOW_BUFFER
#define SPH_DYNAMIC_WINDOW_BUFFER


#include "SPlisHSPlasH\Vector.h"
#include "DFSPH_c_arrays_structure.h"

using namespace SPH;

//don't call that
void handle_fluid_boundries_cuda(SPH::DFSPHCData& data, bool loading = false, Vector3d movement=Vector3d(0,0,0));

#endif //DFSPH_STATIC_VAR_STRUCT