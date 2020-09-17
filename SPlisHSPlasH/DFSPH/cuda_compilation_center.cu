#define REMOVAL_TAG  25000000
#define TAG_REMOVAL  25000000
#define TAG_ACTIVE 1


#include "basic_kernels_cuda.cu"
#include "DFSPH_core_cuda.cu"
#include "DFSPH_memory_management_cuda.cu"
#include "DFSPH_rendering_cuda.cu"
#include "DFSPH_static_variables_structure_cuda.cu"
#include "BufferFluidSurface.cu"
#include "BorderHeightMap.cu"
#include "SPH_other_systems_cuda.cu"
#include "SPH_dynamic_window_buffer.cu"
#include "RestFluidLoader.cu"

//test2