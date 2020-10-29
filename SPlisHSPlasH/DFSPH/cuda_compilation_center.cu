#define REMOVAL_TAG  25000000
#define TAG_REMOVAL  25000000
#define TAG_REMOVAL_CANDIDATE  12500000
#define TAG_ACTIVE 1
#define TAG_ACTIVE_NEIGHBORS 2
#define TAG_ACTIVE_NEIGHBORS_ORDER_2 3
#define TAG_SAVE 4
#define TAG_1 1001
#define TAG_2 1002
#define TAG_3 1003
#define TAG_AIR 30000
#define TAG_AIR_ACTIVE_NEIGHBORS 30002

#define TAG_UNTAGGED 10000


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